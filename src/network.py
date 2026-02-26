"""
Network topology and DeGroot consensus.

Builds an SBM (Stochastic Block Model) graph with 24 nodes in 8 blocks (Block order:
  0: left
  1: center_left
  2: center_right
  3: right
  4: civil_liberties_advocate
  5: tech_utopian
  6: nationalist_populist
  7: conspiracy_theorist). Edges follow homophily: same-block nodes connect more, adjacent blocks
less, opposite ends rarely. Also provides DeGroot consensus for scalar opinions.
"""

from typing import Optional

import networkx as nx
import numpy as np


SBM_BLOCKS = (
    "left",                     # 0
    "center_left",              # 1
    "center_right",             # 2
    "right",                    # 3
    "civil_liberties_advocate", # 4
    "tech_utopian",             # 5
    "nationalist_populist",     # 6
    "conspiracy_theorist",      # 7
)
SBM_SIZES = [3, 3, 3, 3, 3, 3, 3, 3]  # 24 nodes total
# Connection probability matrix (8×8).
# Rows/cols follow the block order above.
SBM_PROBS = [
    #  left   c_lft  c_rgt  right  civil  tech   pop    conspi
    [0.70,  0.40,  0.05,  0.01,  0.25,  0.03,  0.02,  0.01],  # 0: left
    [0.40,  0.70,  0.25,  0.05,  0.30,  0.10,  0.03,  0.01],  # 1: center_left
    [0.05,  0.25,  0.70,  0.40,  0.20,  0.35,  0.10,  0.02],  # 2: center_right
    [0.01,  0.05,  0.40,  0.70,  0.10,  0.30,  0.35,  0.05],  # 3: right
    [0.25,  0.30,  0.20,  0.10,  0.70,  0.15,  0.10,  0.02],  # 4: civil_liberties_advocate
    [0.03,  0.10,  0.35,  0.30,  0.15,  0.70,  0.20,  0.05],  # 5: tech_utopian
    [0.02,  0.03,  0.10,  0.35,  0.10,  0.20,  0.70,  0.40],  # 6: nationalist_populist
    [0.01,  0.01,  0.02,  0.05,  0.02,  0.05,  0.40,  0.70],  # 7: conspiracy_theorist
]



def create_graph(seed: Optional[int] = None) -> nx.Graph:
    """
    Create SBM graph with 24 nodes in 8 blocks.
    Returns:
        NetworkX Graph. Each node has a 'block' (int) and 'block_name' (str) attribute.
    """
    G = nx.stochastic_block_model(SBM_SIZES, SBM_PROBS, seed=seed)
    block_map = {}
    block_name_map = {}
    for b, size in enumerate(SBM_SIZES):
        start = sum(SBM_SIZES[:b])
        for i in range(size):
            block_map[start + i] = b
            block_name_map[start + i] = SBM_BLOCKS[b]
    nx.set_node_attributes(G, block_map, "block")
    nx.set_node_attributes(G, block_name_map, "block_name")
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