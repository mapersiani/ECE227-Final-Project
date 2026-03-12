"""
Semantic measurement via SBERT embeddings.

Converts opinion text to vectors and computes semantic variance (spread from centroid).
Model is cached after first load. Suppresses Hugging Face / transformers verbose output.
"""

import logging
import os
import warnings
from pathlib import Path
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
_persona_proto_mat: Optional[np.ndarray] = None
_persona_side_labels: Optional[list[str]] = None


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


def _ensure_persona_prototypes() -> tuple[np.ndarray, list[str]]:
    """Load/cache persona prototype embeddings and their side labels."""
    global _persona_proto_mat, _persona_side_labels

    def side_from_name(name: str) -> str:
        token = str(name).split("_", 1)[0].strip().lower()
        if token == "republican":
            return "republican"
        # Collapse all non-republican personas into democracy for binary camp analysis.
        return "democracy"

    model = _get_model(show_progress=False)
    if _persona_proto_mat is None or _persona_side_labels is None:
        nodes_path = Path(__file__).resolve().parent / "nodes.json"
        from src.nodes_data import load_nodes_data

        nodes = load_nodes_data(nodes_path)
        names = [str(n.get("name", "")) for n in nodes]
        texts = [str(n.get("initial") or n.get("prompt") or "neutral policy opinion") for n in nodes]
        _persona_side_labels = [side_from_name(n) for n in names]
        _persona_proto_mat = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return _persona_proto_mat, _persona_side_labels


def classify_side_labels(embeddings: np.ndarray) -> list[str]:
    """
    Classify each embedding to one of two camps via nearest persona prototype.
    Returns a side label for each embedding in order.
    """
    if embeddings.size == 0:
        return []

    proto_mat, side_labels = _ensure_persona_prototypes()
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    proto_norm = proto_mat / (np.linalg.norm(proto_mat, axis=1, keepdims=True) + 1e-8)
    sims = emb_norm @ proto_norm.T
    nearest_idx = np.argmax(sims, axis=1)
    return [side_labels[int(j)] for j in nearest_idx]


def classify_sides(embeddings: np.ndarray) -> dict[str, int]:
    """
    Classify each embedding into one of two camps via nearest persona prototype.
    Returns side -> count for the given timestep.
    """
    counts = {"democracy": 0, "republican": 0}
    for side in classify_side_labels(embeddings):
        if side in counts:
            counts[side] += 1
    return counts
