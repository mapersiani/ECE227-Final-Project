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

from src.config import PERSONA_BLOCKS, side_from_name
from src.load_nodes import load_nodes

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer

_model_cache: Optional["SentenceTransformer"] = None
# Cache prototypes per persona_set: persona_set -> (proto_mat, side_labels)
_persona_proto_cache: dict[str, tuple[np.ndarray, list[str]]] = {}


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


def pairwise_cosine_distances(embeddings: np.ndarray) -> np.ndarray:
    """Return (N, N) cosine distance matrix."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    normed = embeddings / norms
    cos_sim = normed @ normed.T
    return 1.0 - np.clip(cos_sim, -1, 1)


def opinion_polarization(embeddings: np.ndarray, blocks: list[int]) -> float:
    """
    Between-block vs within-block semantic distance ratio.
    >1 means opinions are more different across ideological blocks than within them.
    """
    blocks_arr = np.array(blocks)
    unique = np.unique(blocks_arr)
    dist = pairwise_cosine_distances(embeddings)

    within, between = [], []
    n = len(blocks_arr)
    for i in range(n):
        for j in range(i + 1, n):
            if blocks_arr[i] == blocks_arr[j]:
                within.append(dist[i, j])
            else:
                between.append(dist[i, j])

    w = float(np.mean(within)) if within else 0.0
    b = float(np.mean(between)) if between else 0.0
    return b / (w + 1e-9)


def _side_from_name(name: str) -> str:
    s = side_from_name(name)
    return s if s in PERSONA_BLOCKS else "other"


def _ensure_persona_prototypes(persona_set: str = "personas") -> tuple[np.ndarray, list[str]]:
    global _persona_proto_cache
    model = _get_model(show_progress=False)
    if persona_set not in _persona_proto_cache:
        nodes = load_nodes(persona_set)
        names = [n.get("name", "") for n in nodes]
        texts = [n.get("initial") or n.get("prompt") or "" for n in nodes]
        side_labels = [_side_from_name(n) for n in names]
        proto_mat = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        _persona_proto_cache[persona_set] = (proto_mat, side_labels)
    return _persona_proto_cache[persona_set]


def classify_side_labels(
    embeddings: np.ndarray,
    persona_set: str = "personas",
) -> list[str]:
    """
    Classify each embedding into one coarse side label.
    """
    proto_mat, side_labels = _ensure_persona_prototypes(persona_set)

    emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    proto_norm = proto_mat / (np.linalg.norm(proto_mat, axis=1, keepdims=True) + 1e-8)
    sims = emb_norm @ proto_norm.T
    nn = np.argmax(sims, axis=1)

    labels: list[str] = []
    for j in nn:
        side = side_labels[int(j)]
        labels.append(side if side in PERSONA_BLOCKS else "other")
    return labels


def classify_sides(
    embeddings: np.ndarray,
    persona_set: str = "personas",
) -> dict[str, int]:
    """
    Classify each embedding into one of the coarse sides (democrat, republican, independent).

    Method: nearest-neighbor (cosine similarity) to persona prototypes built from
    the selected nodes file. After assigning each agent to the closest persona,
    we map that persona to its side using the persona `name` prefix (party_firstname_lastname).

    Returns:
        Dict side -> count at this timestep.
    """
    counts = {k: 0 for k in PERSONA_BLOCKS}
    for side in classify_side_labels(embeddings, persona_set=persona_set):
        if side in counts:
            counts[side] += 1
    return counts


def mean_persona_drift(embeddings: np.ndarray, prototypes: np.ndarray) -> float:
    """
    Mean cosine distance (1 - similarity) between current embeddings and their
    initial prototypes (t=0 vectors).
    """
    if embeddings.shape != prototypes.shape:
        return 0.0
    norm_e = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    norm_p = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-9)
    cos_sim = np.sum(norm_e * norm_p, axis=1)
    drift = 1.0 - np.clip(cos_sim, -1, 1)
    return float(np.mean(drift))
