"""
Semantic measurement via SBERT embeddings.

Converts opinion text to vectors and computes semantic variance (spread from centroid).
Model is cached after first load. Suppresses Hugging Face / transformers verbose output.
"""

import logging
import os
import warnings
from typing import Optional

import numpy as np

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer

_model_cache: Optional["SentenceTransformer"] = None


def _get_model(show_progress: bool = True) -> "SentenceTransformer":
    """Load SBERT model (cached). Suppresses tqdm during load."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    if show_progress:
        print("Loading SBERT model...", end=" ", flush=True)
    old = os.environ.pop("TQDM_DISABLE", None)
    os.environ["TQDM_DISABLE"] = "1"
    try:
        _model_cache = SentenceTransformer("all-MiniLM-L6-v2")
    finally:
        os.environ.pop("TQDM_DISABLE", None)
        if old is not None:
            os.environ["TQDM_DISABLE"] = old
    if show_progress:
        print("done.")
    return _model_cache


def embed_opinions(
    opinions: list[str],
    model: Optional["SentenceTransformer"] = None,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """
    Map opinion strings to SBERT embedding vectors.

    Returns:
        Array of shape (n_opinions, embedding_dim).
    """
    if model is None:
        model = _get_model()
    return model.encode(opinions, convert_to_numpy=True, show_progress_bar=show_progress_bar)


def semantic_variance(embeddings: np.ndarray) -> float:
    """
    Mean squared distance from centroid. Higher = more polarized/diverse opinions.
    """
    centroid = embeddings.mean(axis=0)
    return float(np.mean(np.sum((embeddings - centroid) ** 2, axis=1)))
