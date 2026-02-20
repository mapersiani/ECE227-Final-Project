"""
Network topology and DeGroot consensus.

Builds an SBM (Stochastic Block Model) graph with 20 nodes in 4 blocks (left, center left,
center right, right). Edges follow homophily: same-block nodes connect more, adjacent blocks
less, opposite ends rarely. Also provides DeGroot consensus for scalar opinions.
"""

from typing import Optional

import networkx as nx
import numpy as np

# Block order: left (0), center left (1), center right (2), right (3)
SBM_BLOCKS = ("left", "center_left", "center_right", "right")
SBM_SIZES = [5, 5, 5, 5]  # 20 nodes total
SBM_PROBS = [
    [0.5, 0.25, 0.05, 0.01],
    [0.25, 0.5, 0.25, 0.05],
    [0.05, 0.25, 0.5, 0.25],
    [0.01, 0.05, 0.25, 0.5],
]


def create_graph(seed: Optional[int] = None) -> nx.Graph:
    """
    Create SBM graph with 20 nodes in 4 blocks.

    Returns:
        NetworkX Graph. Each node has a 'block' attribute (0=left … 3=right).
    """
    G = nx.stochastic_block_model(SBM_SIZES, SBM_PROBS, seed=seed)
    block_map = {}
    for b, size in enumerate(SBM_SIZES):
        start = sum(SBM_SIZES[:b])
        for i in range(size):
            block_map[start + i] = b
    nx.set_node_attributes(G, block_map, "block")
    return G


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
