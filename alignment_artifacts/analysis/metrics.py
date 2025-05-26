"""Metrics computation for artifact analysis."""

import numpy as np
from typing import Dict, Optional


def compute_category_metrics(
    natural_activations: np.ndarray,
    artifact_activations: np.ndarray
) -> Optional[Dict[str, float]]:
    """
    Compute Cohen's d and classification accuracy for a category.
    
    Args:
        natural_activations: Array of natural prompt activations
        artifact_activations: Array of artifact prompt activations
    
    Returns:
        Dict with metrics or None if insufficient data
    """
    if len(natural_activations) == 0 or len(artifact_activations) == 0:
        return None
    
    # Direction vector
    nat_mean = natural_activations.mean(axis=0)
    art_mean = artifact_activations.mean(axis=0)
    direction = art_mean - nat_mean
    
    # Cohen's d
    nat_std = natural_activations.std(axis=0).mean()
    art_std = artifact_activations.std(axis=0).mean()
    pooled_std = np.sqrt((nat_std + art_std) / 2)
    cohens_d = np.linalg.norm(direction) / (pooled_std + 1e-9)
    
    # Classification accuracy
    nat_proj = natural_activations @ direction
    art_proj = artifact_activations @ direction
    
    if len(np.unique(np.concatenate([nat_proj, art_proj]))) < 2:
        accuracy = 0.5
    else:
        threshold = (nat_proj.mean() + art_proj.mean()) / 2
        correct = (nat_proj < threshold).sum() + (art_proj > threshold).sum()
        accuracy = correct / (len(nat_proj) + len(art_proj))
    
    return {
        'cohens_d': float(cohens_d),
        'accuracy': float(accuracy),
        'n_natural': len(natural_activations),
        'n_artifact': len(artifact_activations),
        'direction_norm': float(np.linalg.norm(direction)),
        'mean_separation': float(art_proj.mean() - nat_proj.mean())
    }