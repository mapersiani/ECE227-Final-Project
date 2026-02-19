"""Semantic measurement via SBERT embeddings."""

from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer


def _get_model() -> SentenceTransformer:
    """Load SBERT model (cached after first call)."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_opinions(opinions: list[str], model: Optional[SentenceTransformer] = None) -> np.ndarray:
    """
    Map opinion texts to embedding vectors using SBERT.

    Args:
        opinions: List of opinion strings
        model: Optional pre-loaded model (loads default if None)

    Returns:
        Array of shape (n_opinions, embedding_dim)
    """
    if model is None:
        model = _get_model()
    return model.encode(opinions, convert_to_numpy=True)


def semantic_variance(embeddings: np.ndarray) -> float:
    """
    Compute semantic variance (mean squared distance from centroid).

    Args:
        embeddings: Array of shape (n, dim)

    Returns:
        Semantic variance scalar
    """
    centroid = embeddings.mean(axis=0)
    return float(np.mean(np.sum((embeddings - centroid) ** 2, axis=1)))


def semantic_polarization(embeddings: np.ndarray) -> float:
    """
    Semantic polarization: standard deviation of pairwise cosine distances
    (alternative: mean pairwise distance). Higher = more polarized.

    Here we use the mean pairwise Euclidean distance for interpretability.
    """
    n = len(embeddings)
    if n < 2:
        return 0.0
    diffs = embeddings[:, None, :] - embeddings[None, :, :]
    norms = np.sqrt(np.sum(diffs ** 2, axis=2))
    return float(np.mean(norms[np.triu_indices(n, k=1)]))


def embedding_distances_from_centroid(embeddings: np.ndarray) -> np.ndarray:
    """Per-opinion distance from centroid (for analysis)."""
    centroid = embeddings.mean(axis=0)
    return np.sqrt(np.sum((embeddings - centroid) ** 2, axis=1))
