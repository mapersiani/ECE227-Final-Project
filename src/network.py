# src/network.py
"""
Network topology and DeGroot consensus (Erdős-Rényi baseline).

Builds an Erdős-Rényi (ER) graph on the personas defined in ``nodes.json``.
Each node corresponds to one persona (left, center_left, center_right, right).

Upgrades:
- Optionally parameterize ER by avg_degree (more comparable than raw p)
- Avoid isolated nodes (so every agent can update)
- Provide basic graph stats for logging/debugging
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


def graph_stats(G: nx.Graph) -> dict[str, float | int]:
    degs = [d for _, d in G.degree()]
    isolates = sum(1 for d in degs if d == 0)
    return {
        "nodes": int(G.number_of_nodes()),
        "edges": int(G.number_of_edges()),
        "min_deg": int(min(degs)) if degs else 0,
        "mean_deg": float(np.mean(degs)) if degs else 0.0,
        "max_deg": int(max(degs)) if degs else 0,
        "isolates": int(isolates),
        "connected_components": int(nx.number_connected_components(G)) if G.number_of_nodes() > 0 else 0,
    }


def create_graph(
    edge_prob: float = 0.15,
    seed: Optional[int] = None,
    avg_degree: Optional[float] = None,
    ensure_no_isolates: bool = True,
) -> nx.Graph:
    """
    Create an Erdős–Rényi graph on the personas in nodes.json.

    Args:
        edge_prob: ER edge probability p (0–1). Ignored if avg_degree is provided.
        seed: Random seed for reproducibility.
        avg_degree: If set, uses p = avg_degree/(n-1) for comparability across n.
        ensure_no_isolates: If True, connect any isolated node to a random other node.

    Returns:
        NetworkX Graph with one node per persona and attributes:
        - name, prompt, style, initial_text, side
    """
    nodes = load_nodes()
    n = len(nodes)

    if n <= 1:
        G = nx.empty_graph(n=n)
    else:
        if avg_degree is not None:
            # expected degree k => p = k/(n-1)
            edge_prob = float(avg_degree) / float(n - 1)
            edge_prob = max(0.0, min(1.0, edge_prob))

        G = nx.erdos_renyi_graph(n=n, p=edge_prob, seed=seed)

        if ensure_no_isolates and n > 1:
            rng = np.random.default_rng(seed)
            isolates = [i for i in G.nodes() if G.degree(i) == 0]
            for i in isolates:
                # connect i to a random node != i
                j = int(rng.integers(0, n - 1))
                if j >= i:
                    j += 1
                G.add_edge(i, j)

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
    """
    W = degroot_weights(G)
    history: list[np.ndarray] = [initial_opinions.copy()]
    x = initial_opinions.copy()
    for _ in range(steps):
        x = W @ x
        history.append(x.copy())
    return history
