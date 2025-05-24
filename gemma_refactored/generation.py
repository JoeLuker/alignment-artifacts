"""
Text generation utilities with activation capture support.
"""

import time
from typing import Dict, Generator, List, Optional, Set, Tuple, Union
import mlx.core as mx
import mlx.nn as nn

from .activation_capture import activation_store, save_activations
from .model_architecture import create_attention_mask, create_additive_causal_mask


class KVCache:
    """Key-Value cache for efficient generation."""
    
    def __init__(self, head_dim, step=256, max_size: Optional[int] = None):
        self.head_dim = head_dim
        self.step = step
        self.max_size = max_size  # For sliding window
        self.keys = None
        self.values = None
        self.offset = 0
        self.batch_size = None
        self.dtype = None

    def update_and_fetch(self, keys, values):
        """Update cache and return full context."""
        if self.keys is None:
            self.keys = keys
            self.values = values
            self.batch_size = keys.shape[0]
            self.dtype = keys.dtype

        B, H, L, D = keys.shape  # Get current shape
        current_len = self.keys.shape[2] if self.keys is not None else 0
        needed_len = self.offset + L

        # Reallocate if needed
        if self.keys is None or needed_len > current_len:
            alloc_len = ((needed_len + self.step - 1) // self.step) * self.step
            new_k = mx.zeros((B, H, alloc_len, D), self.dtype)
            new_v = mx.zeros((B, H, alloc_len, D), self.dtype)
            if self.keys is not None:
                copy_len = min(self.offset, current_len)
                mx.eval(self.keys, self.values)  # Sync before copy
                new_k[..., :copy_len, :] = self.keys[..., :copy_len, :]
                new_v[..., :copy_len, :] = self.values[..., :copy_len, :]
            self.keys, self.values = new_k, new_v

        # Sliding window trim simulation
        if self.max_size is not None and needed_len > self.max_size:
            num_to_discard = needed_len - self.max_size
            num_to_keep = self.offset - num_to_discard
            if num_to_keep > 0:
                mx.eval(self.keys, self.values)  # Sync before shift
                self.keys[..., :num_to_keep, :] = self.keys[..., num_to_discard : self.offset, :]
                self.values[..., :num_to_keep, :] = self.values[..., num_to_discard : self.offset, :]
                self.offset = num_to_keep
            else:
                self.offset = 0  # Discarded everything

        # Add new keys/values
        write_offset = self.offset
        mx.eval(keys, values, self.keys, self.values)  # Sync before write
        self.keys[..., write_offset : write_offset + L, :] = keys
        self.values[..., write_offset : write_offset + L, :] = values
        self.offset += L

        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def is_trimmable(self): 
        return True
    
    def trim(self, n): 
        n = min(self.offset, n)
        self.offset -= n
        return n


def top_p_sampling(logits, top_p, temp=1.0):
    """Sample from top-p (nucleus) sampling distribution."""
    logits = logits.astype(mx.float32)  # Use float32 for stability
    if temp == 0:  # Handle deterministic case first
        return mx.argmax(logits, axis=-1, keepdims=True)
    if temp != 1.0: 
        logits = logits / temp

    probs = mx.softmax(logits, axis=-1)
    sorted_indices = mx.argsort(-probs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
    mask = cumulative_probs > top_p
    # Shift mask to include the first element that exceeds top_p
    mask = mx.concatenate([mx.zeros_like(mask[..., :1]), mask[..., :-1]], axis=-1)
    probs_filtered = mx.where(mask, mx.zeros_like(sorted_probs), sorted_probs)
    probs_normalized = probs_filtered / (mx.sum(probs_filtered, axis=-1, keepdims=True) + 1e-9)
    indices = mx.random.categorical(mx.log(probs_normalized + 1e-9))
    selected_indices = mx.take_along_axis(sorted_indices, indices[..., None], axis=-1)
    return selected_indices.astype(mx.int32)


def apply_repetition_penalty(logits: mx.array, generated_tokens: List[List[int]], penalty: float):
    """
    Apply repetition penalty to specific logits based on the given context.
    
    Args:
        logits: [batch_size, vocab_size] logits to penalize
        generated_tokens: List[List[int]] - list of token sequences for each batch item
        penalty: float - penalty factor (>1.0 to discourage repetition)
    """
    if not generated_tokens or penalty == 1.0:
        return logits

    batch_size, vocab_size = logits.shape
    penalized_logits = mx.array(logits)  # Work on a copy

    # Process each sequence in the batch
    for i in range(min(batch_size, len(generated_tokens))):
        if not generated_tokens[i]:
            continue
            
        # Get unique tokens for this sequence
        unique_tokens = list(set(generated_tokens[i]))
        valid_indices = [idx for idx in unique_tokens if 0 <= idx < vocab_size]
        
        if valid_indices:
            for idx in valid_indices:
                selected_logit = penalized_logits[i, idx]
                # Apply penalty (> 0 divide, <= 0 multiply)
                penalized_value = mx.where(selected_logit > 0, 
                                         selected_logit / penalty, 
                                         selected_logit * penalty)
                penalized_logits[i, idx] = penalized_value

    return penalized_logits


def generate_step(
    prompts: mx.array,
    model: nn.Module,
    temp: float = 0.0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    top_p: float = 1.0,
    logit_bias: Optional[Dict[int, float]] = None,
    save_activations_dir: Optional[str] = None,
    save_compress: bool = True,
    eos_token_ids: Optional[Set] = None,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing token ids, modified to capture activations.
    Yields: Tuple[mx.array, mx.array]: (token_ids, probabilities)
    """
    # Activation Setup
    activation_store.reset()
    activation_store.enable()
    B, S = prompts.shape
    eos_token_ids = eos_token_ids or set()

    # Sampling Function (Internal, with activation capture)
    def sample(logits: mx.array, step_name: str) -> Tuple[mx.array, mx.array]:
        # Register inputs to sample function
        activation_store.register(f"{step_name}.sample.input_logits", logits)
        current_logits = logits

        # Apply logit bias
        if logit_bias:
            indices = mx.array(list(logit_bias.keys()), dtype=mx.int32)
            values = mx.array(list(logit_bias.values()), dtype=current_logits.dtype)
            vocab_size = current_logits.shape[-1]
            valid_mask = (indices >= 0) & (indices < vocab_size)
            valid_indices = indices[valid_mask]
            valid_values = values[valid_mask]
            if valid_indices.size > 0:
                bias_vector = mx.zeros_like(current_logits)
                for idx, val in zip(valid_indices.tolist(), valid_values.tolist()):
                    bias_vector[:, idx] += val
                current_logits = current_logits + bias_vector
                activation_store.register(f"{step_name}.sample.biased_logits", current_logits)

        # Calculate softmax probabilities (use float32 for stability)
        softmax_probs = mx.softmax(current_logits.astype(mx.float32), axis=-1)
        activation_store.register(f"{step_name}.sample.softmax_probs", softmax_probs)

        # Determine next token
        if temp == 0:
            tokens = mx.argmax(current_logits, axis=-1, keepdims=True)
        else:
            scaled_logits = current_logits / temp
            activation_store.register(f"{step_name}.sample.scaled_logits", scaled_logits)
            if top_p > 0.0 and top_p < 1.0:  # Ensure top_p is within (0, 1) for top-p sampling
                tokens = top_p_sampling(scaled_logits, top_p, temp=1.0)  # Temp already applied
            else:
                tokens = mx.random.categorical(scaled_logits)
                tokens = mx.expand_dims(tokens, axis=-1)

        activation_store.register(f"{step_name}.sample.output_token", tokens)
        # Get probability of the chosen token
        probs = mx.take_along_axis(softmax_probs, tokens, axis=-1).squeeze(axis=-1)
        activation_store.register(f"{step_name}.sample.output_token_prob", probs)
        return tokens.astype(mx.int32), probs

    # Prefill
    y = prompts
    activation_store.register("generate.initial_prompt", y)

    # Setup KV cache
    if hasattr(model, 'make_cache') and callable(model.make_cache):
        print("Creating KV cache...")
        cache = model.make_cache(batch_size=B)
    else:
        print("Warning: Model has no make_cache method. KV cache disabled.")
        cache = None

    # Repetition penalty context setup
    repetition_context = [[] for _ in range(B)]  # List of lists for batch
    rep_context_size = -1
    if repetition_penalty is not None and repetition_penalty != 1.0:
        rep_context_size = repetition_context_size if repetition_context_size is not None and repetition_context_size > 0 else S
        for i in range(B):
            start_idx = max(0, S - rep_context_size)
            repetition_context[i] = list(prompts[i, start_idx:].tolist())

    # Prefill forward pass
    print(f"Processing prefill (prompt shape: {y.shape})...")
    start_prefill = time.time()
    prefill_mask = create_attention_mask(y, cache=None, return_array=True, dtype=y.dtype)
    activation_store.register("generate.prefill.input_mask", prefill_mask)
    prefill_logits = model(y, cache=cache, mask=prefill_mask)
    mx.eval(prefill_logits)  # Sync after prefill
    end_prefill = time.time()
    print(f"Prefill computation time: {end_prefill - start_prefill:.3f}s")

    # Capture prefill activations
    activation_store.register("generate.prefill.full_logits", prefill_logits)
    last_prefill_logits = prefill_logits[:, -1, :]
    activation_store.register("generate.prefill.last_token_logits", last_prefill_logits)

    # Apply repetition penalty for first token
    penalized_prefill_logits = last_prefill_logits
    if repetition_penalty is not None and repetition_penalty != 1.0:
        penalized_prefill_logits = apply_repetition_penalty(last_prefill_logits, repetition_context, repetition_penalty)
        activation_store.register("generate.prefill.penalized_logits", penalized_prefill_logits)
    else:
        activation_store.register("generate.prefill.penalized_logits", last_prefill_logits)

    # Sample first token
    y, p = sample(penalized_prefill_logits, "generate.prefill")

    # Update repetition context
    if repetition_penalty is not None and repetition_penalty != 1.0:
        for i in range(B):
            token_item = y[i, 0].item()
            repetition_context[i].append(token_item)
            if rep_context_size > 0 and len(repetition_context[i]) > rep_context_size:
                repetition_context[i] = repetition_context[i][-rep_context_size:]

    # Save prefill activations if requested
    if save_activations_dir:
        print("Saving prefill activations...")
        save_activations(activation_store.get_captured_activations(), save_activations_dir, step="prefill", compress=save_compress)

    # Generation Loop
    step_count = 0
    finished = [False] * B

    while True:
        # Yield the token generated in the previous step
        mx.eval(y, p)  # Ensure token/prob are computed before yielding
        yield y, p

        # Check completion status
        all_finished = True
        for i in range(B):
            if not finished[i]:
                if y[i, 0].item() in eos_token_ids: 
                    finished[i] = True
                else: 
                    all_finished = False
        if all_finished: 
            print("\nAll sequences finished.")
            break

        # Reset activations for the new step
        activation_store.reset()
        step_name = f"generate.step_{step_count}"
        step_input_token = y  # Input is the token just yielded
        activation_store.register(f"{step_name}.input_token", step_input_token)

        # Forward pass for the single token
        step_mask = create_attention_mask(step_input_token, cache=cache, return_array=True, dtype=step_input_token.dtype)
        activation_store.register(f"{step_name}.input_mask", step_mask)
        step_logits = model(step_input_token, cache=cache, mask=step_mask)
        last_token_logits = step_logits[:, -1, :]  # Shape [B, V]
        activation_store.register(f"{step_name}.last_token_logits", last_token_logits)

        # Apply repetition penalty
        penalized_logits = last_token_logits
        if repetition_penalty is not None and repetition_penalty != 1.0:
            penalized_logits = apply_repetition_penalty(last_token_logits, repetition_context, repetition_penalty)
            activation_store.register(f"{step_name}.penalized_logits", penalized_logits)
        else:
            activation_store.register(f"{step_name}.penalized_logits", last_token_logits)

        # Sample next token
        y, p = sample(penalized_logits, step_name)

        # Update repetition context
        if repetition_penalty is not None and repetition_penalty != 1.0:
            for i in range(B):
                if not finished[i]:
                    token_item = y[i, 0].item()
                    repetition_context[i].append(token_item)
                    if rep_context_size > 0 and len(repetition_context[i]) > rep_context_size:
                        repetition_context[i] = repetition_context[i][-rep_context_size:]

        # Save step activations if requested
        if save_activations_dir:
            save_activations(activation_store.get_captured_activations(), save_activations_dir, step=step_count, compress=save_compress)

        step_count += 1


def batch_generate(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    repetition_context_size: int = 20,
    logit_bias: Optional[Dict[int, float]] = None,
    save_activations_dir: Optional[str] = None,
    save_compress: bool = True,
    verbose: bool = False,
) -> List[Dict[str, Union[str, List[int]]]]:
    """Generate text for multiple prompts with activation capture."""
    
    # Tokenize prompts
    tokenized_prompts = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        tokenized_prompts.append(tokens)
    
    # Pad to same length
    max_len = max(len(tokens) for tokens in tokenized_prompts)
    padded_prompts = []
    for tokens in tokenized_prompts:
        if len(tokens) < max_len:
            # Pad with tokenizer pad token or 0
            pad_token = getattr(tokenizer, 'pad_token_id', 0)
            tokens = [pad_token] * (max_len - len(tokens)) + tokens
        padded_prompts.append(tokens)
    
    prompt_tokens = mx.array(padded_prompts)
    
    # Get EOS token IDs
    eos_token_ids = set()
    if hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        eos_token_ids.add(tokenizer.eos_token_id)
    
    # Generate
    results = []
    generated_tokens_list = [[] for _ in range(len(prompts))]
    
    generator = generate_step(
        prompts=prompt_tokens,
        model=model,
        temp=temperature,
        repetition_penalty=repetition_penalty,
        repetition_context_size=repetition_context_size,
        top_p=top_p,
        logit_bias=logit_bias,
        save_activations_dir=save_activations_dir,
        save_compress=save_compress,
        eos_token_ids=eos_token_ids,
    )
    
    for step, (tokens, probs) in enumerate(generator):
        if step >= max_tokens:
            break
            
        # Store generated tokens
        for i in range(len(prompts)):
            token = tokens[i, 0].item()
            generated_tokens_list[i].append(token)
            
        if verbose:
            print(f"Step {step}: Generated tokens {tokens.tolist()}")
    
    # Decode results
    for i, (prompt, generated_tokens) in enumerate(zip(prompts, generated_tokens_list)):
        try:
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        except Exception as e:
            print(f"Warning: Failed to decode tokens for prompt {i}: {e}")
            generated_text = f"<decoding_error: {generated_tokens}>"
        
        results.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "generated_tokens": generated_tokens,
            "num_tokens": len(generated_tokens)
        })
    
    return results