"""
Alignment Artifacts Library

A library for analyzing and suppressing alignment artifacts in language models.
"""

from .core.library import AlignmentArtifacts
from .core.api import suppress_prompt, analyze_model

__version__ = "0.1.0"
__all__ = ["AlignmentArtifacts", "suppress_prompt", "analyze_model"]