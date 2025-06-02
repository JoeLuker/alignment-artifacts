"""Simple API functions for the alignment artifacts library."""

from typing import Optional
from .library import AlignmentArtifacts


def suppress_prompt(
    prompt: str, model: Optional[str] = None, scale: float = 1.0, **kwargs
):
    """
    Quick function to suppress a single prompt.

    Args:
        prompt: The prompt to process
        model: Model name (defaults to gemma-3-1b-it-qat-4bit)
        scale: Suppression strength (0=none, higher=stronger)
        **kwargs: Additional arguments passed to suppress_and_generate

    Returns:
        Dict with 'baseline' and 'suppressed' outputs
    """
    lib = AlignmentArtifacts()
    model = model or "mlx-community/gemma-3-1b-it-qat-4bit"
    return lib.suppress_and_generate(prompt, model, scale, **kwargs)


def analyze_model(model: str):
    """
    Analyze alignment artifacts for a model.

    Args:
        model: Model name to analyze

    Returns:
        Dict with analysis results including best layers
    """
    lib = AlignmentArtifacts()
    return lib.analyze_artifacts(model)
