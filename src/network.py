"""
Network topology and DeGroot consensus.

Provides ER/small-world/scale-free graph builders and DeGroot consensus.
"""

from typing import Optional

import networkx as nx
import numpy as np
def create_erdos_renyi_graph(n: int, p: float = 0.2, seed: Optional[int] = None) -> nx.Graph:
    """
    Create an Erdos-Renyi G(n, p) graph.
    """
    return nx.erdos_renyi_graph(n=n, p=p, seed=seed)


def create_small_world_rgg_graph(
    n: int,
    radius: float = 0.35,
    long_edge_prob: float = 0.03,
    seed: Optional[int] = None,
) -> nx.Graph:
    """
    Small-world graph built from RGG + random long-distance edges.

    1) Start from random geometric graph G_rgg(n, radius).
    2) Add random edges between currently non-adjacent pairs with distance > radius
       using probability long_edge_prob.
    """
    rng = np.random.default_rng(seed)
    G = nx.random_geometric_graph(n=n, radius=radius, seed=seed)
    pos = nx.get_node_attributes(G, "pos")
    nodes = list(G.nodes())
    for idx, u in enumerate(nodes):
        pu = pos[u]
        for v in nodes[idx + 1 :]:
            if G.has_edge(u, v):
                continue
            pv = pos[v]
            dist = float(np.linalg.norm(np.array(pu) - np.array(pv)))
            if dist <= radius:
                continue
            if rng.random() < long_edge_prob:
                G.add_edge(u, v)
    return G


def create_scale_free_graph(n: int, m: int = 2, seed: Optional[int] = None) -> nx.Graph:
    """
    Scale-free graph via Barabasi-Albert preferential attachment.
    """
    m_eff = max(1, min(m, n - 1))
    return nx.barabasi_albert_graph(n=n, m=m_eff, seed=seed)


def degroot_weights(G: nx.Graph) -> np.ndarray:
    """
    Row-stochastic weight matrix for DeGroot. Each node gives equal weight to all neighbors.
    """
    n = G.number_of_nodes()
    W = np.zeros((n, n))
    for i in range(n):
        neighbors = list(G.neighbors(i))
        deg = len(neighbors)
        if deg > 0:
            for j in neighbors:
                W[i, j] = 1.0 / deg
    return W


def run_degroot(G: nx.Graph, initial_opinions: np.ndarray, steps: int = 5) -> list:
    """
    Run classical DeGroot consensus. Opinions converge to network average.

    Args:
        G: Graph
        initial_opinions: 1D array of scalar opinions per node
        steps: Number of update steps

    Returns:
        List of opinion vectors (t=0 through t=steps).
    """
    W = degroot_weights(G)
    history = [initial_opinions.copy()]
    x = initial_opinions.copy()
    for _ in range(steps):
        x = W @ x
        history.append(x.copy())
    return history
