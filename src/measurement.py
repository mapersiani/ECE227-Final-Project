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

from src.config import PERSONA_BLOCKS
from src.network import load_nodes

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


def classify_sides(embeddings: np.ndarray) -> dict[str, int]:
    """
    Classify each embedding into one of the four coarse sides
    (left, center_left, center_right, right).

    Method: nearest-neighbor (cosine similarity) to *persona prototypes* built from
    `nodes.json` initial opinions. After assigning each agent to the closest persona,
    we map that persona to its side using the persona `name` prefix.

    This ensures timestep 0 classification matches the underlying `nodes.json` balance
    (e.g., 9/9/9/9 if `nodes.json` contains 9 personas per side).

    Returns:
        Dict side -> count at this timestep.
    """
    global _persona_proto_mat, _persona_side_labels

    def side_from_name(name: str) -> str:
        for s in PERSONA_BLOCKS:
            if name.startswith(s):
                return s
        return "other"

    model = _get_model(show_progress=False)
    if _persona_proto_mat is None or _persona_side_labels is None:
        nodes = load_nodes()
        names = [n.get("name", "") for n in nodes]
        texts = [n.get("initial") or n.get("prompt") or "" for n in nodes]
        _persona_side_labels = [side_from_name(n) for n in names]
        _persona_proto_mat = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    proto_mat = _persona_proto_mat  # (m, d), m = number of personas in nodes.json
    side_labels = _persona_side_labels

    # Cosine similarity between each embedding and each persona prototype
    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    proto_norm = proto_mat / (np.linalg.norm(proto_mat, axis=1, keepdims=True) + 1e-8)
    sims = emb_norm @ proto_norm.T  # (n_agents, m_personas)
    nn = np.argmax(sims, axis=1)

    counts = {k: 0 for k in PERSONA_BLOCKS}
    for j in nn:
        side = side_labels[int(j)]
        if side in counts:
            counts[side] += 1
    return counts
