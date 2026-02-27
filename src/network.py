"""
Network topology and DeGroot consensus (Erdős-Rényi baseline).

Builds an Erdős-Rényi (ER) graph on the 36 personas defined in ``nodes.json``.
Each node corresponds to one persona (left, center left, center right, right).
Also provides DeGroot consensus for scalar opinions on this fixed node set.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

NODES_PATH = Path(__file__).resolve().parent.parent / "nodes.json"
_NODES_CACHE: list[dict] | None = None


def load_nodes() -> list[dict]:
    """Load persona nodes from nodes.json (cached)."""
    global _NODES_CACHE
    if _NODES_CACHE is None:
        with NODES_PATH.open("r", encoding="utf-8") as f:
            _NODES_CACHE = json.load(f)
    return _NODES_CACHE


def create_graph(edge_prob: float = 0.15, seed: Optional[int] = None) -> nx.Graph:
    """
    Create an Erdős–Rényi graph on the personas in nodes.json.

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
        for s in ("left", "center_left", "center_right", "right"):
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


def export_gephi(G: nx.Graph, out_dir: Path, basename: str) -> tuple[Path, Path]:
    """
    Export the graph for Gephi.

    Writes both:
    - GEXF (preferred by Gephi)
    - GraphML (backup)

    To keep files manageable, exports a trimmed view with only:
    - node label (persona name)
    - side (left/center_left/center_right/right)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(G.edges())
    for n, data in G.nodes(data=True):
        H.nodes[n]["label"] = data.get("name", str(n))
        H.nodes[n]["side"] = data.get("side", "unknown")

    gexf_path = out_dir / f"{basename}.gexf"
    graphml_path = out_dir / f"{basename}.graphml"
    nx.write_gexf(H, gexf_path)
    nx.write_graphml(H, graphml_path)
    return gexf_path, graphml_path


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
