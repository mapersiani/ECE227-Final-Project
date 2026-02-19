"""Network topology: SBM baseline with 4 blocks (left, center left, center right, right)."""

from typing import Optional

import networkx as nx
import numpy as np


# Block order: left (0), center left (1), center right (2), right (3)
SBM_BLOCKS = ("left", "center_left", "center_right", "right")
SBM_SIZES = [5, 5, 5, 5]  # 20 nodes total, 5 per block

# Probability matrix: p[i,j] = edge prob between block i and block j
# Higher within-block, moderate between adjacent blocks, lower across spectrum
SBM_PROBS = [
    [0.5, 0.25, 0.05, 0.01],   # left: strong with center-left, weak with right
    [0.25, 0.5, 0.25, 0.05],   # center left
    [0.05, 0.25, 0.5, 0.25],   # center right
    [0.01, 0.05, 0.25, 0.5],   # right
]


def create_graph(seed: Optional[int] = None) -> nx.Graph:
    """
    Create SBM graph with 20 nodes in 4 blocks: left, center left, center right, right.

    Returns:
        NetworkX Graph. Each node has a 'block' attribute (0=left, 1=center_left, etc.).
    """
    G = nx.stochastic_block_model(
        SBM_SIZES,
        SBM_PROBS,
        seed=seed,
    )
    # Attach block labels to nodes
    block_map = {}
    idx = 0
    for b, size in enumerate(SBM_SIZES):
        for _ in range(size):
            block_map[idx] = b
            idx += 1
    nx.set_node_attributes(G, block_map, "block")
    return G


def degroot_weights(G: nx.Graph) -> np.ndarray:
    """
    Compute row-stochastic weight matrix for DeGroot consensus.
    Each node gives equal weight to all neighbors.
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


def degroot_step(opinions: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Single DeGroot update: x_new = W @ x."""
    return W @ opinions


def run_degroot(
    G: nx.Graph,
    initial_opinions: np.ndarray,
    steps: int = 5,
) -> list:
    """
    Run classical DeGroot consensus with scalar opinions.

    Args:
        G: Graph
        initial_opinions: 1D array of length n (scalar per node)
        steps: Number of steps

    Returns:
        List of opinion vectors (one per timestep, including t=0).
    """
    W = degroot_weights(G)
    history = [initial_opinions.copy()]
    x = initial_opinions.copy()
    for _ in range(steps):
        x = W @ x
        history.append(x.copy())
    return history
