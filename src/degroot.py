"""
DeGroot opinion dynamics baseline.
"""

from typing import Optional

import networkx as nx
import numpy as np


def degroot_weights(G: nx.Graph) -> np.ndarray:
    """
    Compute row-stochastic weight matrix for DeGroot consensus.
    Each node gives equal weight to all neighbors (including itself if self-loops exist, 
    but we will just handle adjacency). Actually, standard DeGroot usually assumes self-loops or 
    equal weighting including self. Let's do equal weight to all neighbors.
    If we want each node to keep some of its own opinion, we can add self-loops to G or do it in the matrix.
    Given the previous diff:
    for i in range(n):
        neighbors = list(G.neighbors(i))
        deg = len(neighbors)
        if deg > 0:
            for j in neighbors:
                W[i, j] = 1.0 / deg
    It didn't explicitly include self, so it averages neighbors. Let's align with the previous code.
    """
    n = G.number_of_nodes()
    W = np.zeros((n, n))
    for i in range(n):
        neighbors = list(G.neighbors(i))
        # Important: usually DeGroot includes oneself. Let's add self to the neighbors list 
        # to prevent singular/bipartite oscillation, though the old code might not have.
        # Let's look at the old code from the git log again:
        # It literally did:
        # neighbors = list(G.neighbors(i))
        # deg = len(neighbors)
        # if deg > 0:
        #    for j in neighbors: W[i,j] = 1.0/deg
        # But this means the node's own opinion is lost unless there are self-loops.
        # To be safe and stable, let's include self.
        neighbors.append(i)
        deg = len(neighbors)
        for j in neighbors:
            W[i, j] = 1.0 / deg
    return W


def run_degroot(
    G: nx.Graph,
    initial_opinions: np.ndarray,
    steps: int = 5,
) -> list[np.ndarray]:
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
