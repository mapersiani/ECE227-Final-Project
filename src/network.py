"""
Network topology and DeGroot consensus (Erdős-Rényi baseline).

Builds an Erdős-Rényi (ER) graph on the 36 personas defined in ``data/nodes.json``.
Each node corresponds to one persona (left, center left, center right, right).
Also provides DeGroot consensus for scalar opinions on this fixed node set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

from src.config import DEFAULT_N, PERSONA_BLOCKS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NODES_PATH = PROJECT_ROOT / "data" / "nodes.json"
_NODES_CACHE: list[dict] | None = None


def load_nodes() -> list[dict]:
    """Load canonical persona nodes from data/nodes.json (cached)."""
    global _NODES_CACHE
    if _NODES_CACHE is None:
        if not NODES_PATH.exists():
            raise FileNotFoundError(f"Could not find canonical nodes file: {NODES_PATH}")
        with NODES_PATH.open("r", encoding="utf-8") as f:
            _NODES_CACHE = json.load(f)
        if len(_NODES_CACHE) != DEFAULT_N:
            raise ValueError(
                f"Expected {DEFAULT_N} nodes in {NODES_PATH.name}, found {len(_NODES_CACHE)}."
            )
    return _NODES_CACHE


def create_graph(edge_prob: float = 0.15, seed: Optional[int] = None) -> nx.Graph:
    """
    Create an Erdős–Rényi graph on the personas in data/nodes.json.

    Args:
        edge_prob: ER edge probability p (0–1)
        seed: Random seed for reproducibility

    Returns:
        NetworkX Graph with one node per persona and attributes:
        - name, prompt, style, initial_text, side (left/center_left/center_right/right)
    """
    nodes = load_nodes()
    n = len(nodes)
    G = nx.erdos_renyi_graph(n=n, p=edge_prob, seed=seed)

    attrs: dict[int, dict] = {}
    for i, node in enumerate(nodes):
        name = node.get("name", f"node_{i}")
        side = "unknown"
        for s in PERSONA_BLOCKS:
            if name.startswith(s):
                side = s
                break
        attrs[i] = {
            "name": name,
            "prompt": node.get("prompt", ""),
            "style": node.get("style", ""),
            "initial_text": node.get("initial", ""),
            "side": side,
        }
    nx.set_node_attributes(G, attrs)
    return G


def degroot_weights(G: nx.Graph) -> np.ndarray:
    """Row-stochastic weight matrix for DeGroot (equal weight on all neighbors)."""
    n = G.number_of_nodes()
    W = np.zeros((n, n))
    for i in range(n):
        neighbors = list(G.neighbors(i))
        deg = len(neighbors)
        if deg > 0:
            for j in neighbors:
                W[i, j] = 1.0 / deg
    return W


def run_degroot(G: nx.Graph, initial_opinions: np.ndarray, steps: int = 5) -> list[np.ndarray]:
    """
    Run classical DeGroot consensus on the given graph.

    Args:
        G: Graph (e.g., ER on 36 personas)
        initial_opinions: 1D array of scalar opinions per node
        steps: Number of update steps

    Returns:
        List of opinion vectors (t=0 through t=steps).
    """
    W = degroot_weights(G)
    history: list[np.ndarray] = [initial_opinions.copy()]
    x = initial_opinions.copy()
    for _ in range(steps):
        x = W @ x
        history.append(x.copy())
    return history
