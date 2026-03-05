"""Semantic polarization metric wrappers."""

from __future__ import annotations

import numpy as np

from src.measurement import classify_sides, embed_opinions, semantic_variance


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed opinion texts."""
    return embed_opinions(texts)


def semantic_spread(embeddings: np.ndarray) -> float:
    """Return semantic variance for embedded opinions."""
    return semantic_variance(embeddings)


def classify_agent_sides(embeddings: np.ndarray) -> dict[str, int]:
    """Classify embedded opinions into coarse side labels."""
    return classify_sides(embeddings)
