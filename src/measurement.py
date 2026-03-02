"""
Semantic measurement and graph metrics.

Opinion/persona embeddings via SBERT → semantic variance, polarization, and drift.
Graph metrics: clustering, modularity, bridging, degree assortativity, and more.
All per-step results are collected into a StepMetrics dataclass for easy analysis.
"""

import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from src.agent import Agent

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

from sentence_transformers import SentenceTransformer

_model_cache: Optional[SentenceTransformer] = None


def _get_model(show_progress: bool = True) -> SentenceTransformer:
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
    opinions: List[str],
    model: Optional[SentenceTransformer] = None,
    show_progress_bar: bool = False,
) -> np.ndarray:
    """Map opinion strings → SBERT embedding matrix (N, D)."""
    if model is None:
        model = _get_model()
    return model.encode(opinions, convert_to_numpy=True, show_progress_bar=show_progress_bar)


def semantic_variance(embeddings: np.ndarray) -> float:
    """Mean squared distance from centroid. Higher = more diverse/polarized opinions."""
    centroid = embeddings.mean(axis=0)
    return float(np.mean(np.sum((embeddings - centroid) ** 2, axis=1)))


def pairwise_cosine_distances(embeddings: np.ndarray) -> np.ndarray:
    """Return (N, N) cosine distance matrix."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    normed = embeddings / norms
    cos_sim = normed @ normed.T
    return 1.0 - np.clip(cos_sim, -1, 1)


def opinion_polarization(embeddings: np.ndarray, blocks: List[int]) -> float:
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


# ─── Graph metrics ─────────────────────────────────────────────────────────────

def compute_graph_metrics(G: nx.Graph, agents: List["Agent"]) -> dict:
    """
    Comprehensive snapshot of graph-level and node-level metrics.

    Structural metrics:
        - avg_clustering: mean local clustering coefficient
        - transitivity: global clustering (fraction of closed triangles)
        - avg_shortest_path: mean geodesic (only for connected graphs)
        - diameter: longest shortest path
        - degree_assortativity: tendency for high-degree nodes to connect to high-degree nodes
        - modularity: quality of block partition (uses 'block' node attribute)
        - bridge_edge_fraction: fraction of edges that are bridges (removal disconnects graph)

    Centrality (per-node, averaged):
        - avg_betweenness: flow of information through nodes
        - avg_eigenvector: influence by connection to influential nodes

    Semantic metrics (requires agents):
        - opinion_variance: spread of current opinions in embedding space
        - persona_drift_mean: mean cosine distance between initial and current persona embeddings
        - persona_drift_std: std of per-node drift
        - opinion_polarization: between-block / within-block opinion distance ratio

    Returns dict of scalar metrics plus 'node_metrics' list (one dict per node).
    """
    n = G.number_of_nodes()
    blocks = [G.nodes[i].get("block", 0) for i in range(n)]
    degrees = dict(G.degree())

    # ── Structural ──
    avg_clustering = nx.average_clustering(G)
    transitivity = nx.transitivity(G)

    try:
        avg_sp = nx.average_shortest_path_length(G)
        diameter = nx.diameter(G)
    except nx.NetworkXError:
        # Not connected — use largest component
        lcc = max(nx.connected_components(G), key=len)
        H = G.subgraph(lcc)
        avg_sp = nx.average_shortest_path_length(H)
        diameter = nx.diameter(H)

    try:
        deg_assort = nx.degree_assortativity_coefficient(G)
    except Exception:
        deg_assort = float("nan")

    # Modularity using block partition
    block_communities = {}
    for i, b in enumerate(blocks):
        block_communities.setdefault(b, set()).add(i)
    communities = list(block_communities.values())
    try:
        modularity = nx.community.modularity(G, communities)
    except Exception:
        modularity = float("nan")

    # Bridge edges (cut edges)
    bridge_edges = set(nx.bridges(G))
    bridge_fraction = len(bridge_edges) / max(G.number_of_edges(), 1)

    # ── Centrality ──
    betweenness = nx.betweenness_centrality(G, normalized=True)
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {i: 0.0 for i in G.nodes()}

    avg_betweenness = float(np.mean(list(betweenness.values())))
    avg_eigenvector = float(np.mean(list(eigenvector.values())))

    # ── Semantic ──
    model = _get_model(show_progress=False)
    current_opinions = [a.current_opinion for a in agents if not a.is_bot]
    opinion_embs = model.encode(current_opinions, convert_to_numpy=True, show_progress_bar=False)
    op_variance = semantic_variance(opinion_embs)

    real_blocks = [blocks[a.node_id] for a in agents if not a.is_bot]
    op_polarization = opinion_polarization(opinion_embs, real_blocks)

    # Persona drift: cosine distance between initial and current persona embeddings
    real_agents = [a for a in agents if not a.is_bot]
    init_personas = [a.initial_persona for a in real_agents]
    curr_personas = [a.persona_prompt for a in real_agents]
    init_embs = model.encode(init_personas, convert_to_numpy=True, show_progress_bar=False)
    curr_embs = model.encode(curr_personas, convert_to_numpy=True, show_progress_bar=False)

    # Per-node cosine drift
    norms_i = np.linalg.norm(init_embs, axis=1, keepdims=True)
    norms_c = np.linalg.norm(curr_embs, axis=1, keepdims=True)
    cos_sim = np.sum(
        (init_embs / np.where(norms_i == 0, 1e-9, norms_i)) *
        (curr_embs / np.where(norms_c == 0, 1e-9, norms_c)),
        axis=1
    )
    persona_drift = 1.0 - np.clip(cos_sim, -1, 1)

    # ── Per-node metrics ──
    node_metrics = []
    for idx, a in enumerate(real_agents):
        i = a.node_id
        node_metrics.append({
            "node_id": i,
            "name": G.nodes[i].get("name", str(i)),
            "block": blocks[i],
            "degree": degrees.get(i, 0),
            "betweenness": betweenness.get(i, 0.0),
            "eigenvector": eigenvector.get(i, 0.0),
            "is_long_range": G.nodes[i].get("long_range", False),
            "persona_drift": float(persona_drift[idx]),
            "opinion_length": len(a.current_opinion),
            "persona_drift_count": a.persona_drift_count(),
        })

    return {
        # Structural
        "avg_clustering": avg_clustering,
        "transitivity": transitivity,
        "avg_shortest_path": avg_sp,
        "diameter": diameter,
        "degree_assortativity": deg_assort,
        "modularity": modularity,
        "bridge_edge_fraction": bridge_fraction,
        # Centrality
        "avg_betweenness": avg_betweenness,
        "avg_eigenvector": avg_eigenvector,
        # Semantic
        "opinion_variance": op_variance,
        "opinion_polarization": op_polarization,
        "persona_drift_mean": float(np.mean(persona_drift)),
        "persona_drift_std": float(np.std(persona_drift)),
        # Raw
        "node_metrics": node_metrics,
    }


@dataclass
class SimulationRecord:
    """
    Full record of a simulation run. One StepSnapshot per timestep.
    Use .to_dataframes() to get pandas DataFrames for analysis.
    """

    topic: str
    steps: int
    graph_params: dict
    step_snapshots: List[dict] = field(default_factory=list)

    def add_step(self, t: int, metrics: dict) -> None:
        self.step_snapshots.append({"t": t, **{k: v for k, v in metrics.items() if k != "node_metrics"}})

    def scalar_series(self, key: str) -> List[float]:
        """Extract a named metric across all timesteps."""
        return [s[key] for s in self.step_snapshots if key in s]

    def node_metrics_at(self, t: int) -> List[dict]:
        """Return per-node metrics dict at timestep t (requires full metrics stored separately)."""
        raise NotImplementedError("Store full_metrics list from simulation to use this.")

    def summary(self) -> dict:
        """Final-state summary of all scalar metrics."""
        if not self.step_snapshots:
            return {}
        last = self.step_snapshots[-1]
        first = self.step_snapshots[0]
        return {
            "topic": self.topic,
            "steps": self.steps,
            **{f"final_{k}": v for k, v in last.items() if k != "t" and isinstance(v, float)},
            "opinion_variance_change": last.get("opinion_variance", 0) - first.get("opinion_variance", 0),
            "persona_drift_change": last.get("persona_drift_mean", 0) - first.get("persona_drift_mean", 0),
        }