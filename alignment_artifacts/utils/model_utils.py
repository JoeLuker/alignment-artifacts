"""Model loading and generation utilities."""

import mlx.core as mx
from typing import Tuple, Optional

# Import from gemma_refactored
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from gemma_refactored.main import load
from gemma_refactored.generation import generate_step
from gemma_refactored.activation_capture import activation_store


def load_model_and_tokenizer(model_name: str) -> Tuple:
    """Load model and tokenizer."""
    return load(model_name)


def format_prompt(prompt: str, tokenizer) -> str:
    """Format prompt with chat template if available."""
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
    return prompt


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 150,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    """Generate text from a prompt."""
    # Tokenize
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_array = mx.array([prompt_tokens])

    # Reset activation store
    activation_store.reset()
    activation_store.enable()

    # Generate
    generated_tokens = []
    # Include both EOS and end_of_turn tokens as stop tokens
    eos_token_ids = set()
    if hasattr(tokenizer, "eos_token_id"):
        eos_token_ids.add(tokenizer.eos_token_id)
    # Add end_of_turn token (ID 106 for Gemma models)
    eos_token_ids.add(106)  # <end_of_turn>

    generator = generate_step(
        prompts=prompt_array,
        model=model,
        temp=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_ids=eos_token_ids,
    )

    for step, (tokens, _) in enumerate(generator):
        if step >= max_tokens:
            break
        token = tokens[0, 0].item()
        generated_tokens.append(token)
        # Stop if we hit end_of_turn token
        if token in eos_token_ids:
            break
        activation_store.reset()  # Reset for next step

    # Decode and return
    full_output = prompt_tokens + generated_tokens
    return tokenizer.decode(full_output)
