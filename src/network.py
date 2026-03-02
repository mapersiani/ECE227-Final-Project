"""
Network topology: Random Geometric Graph (RGG) with long-range connections.

Nodes are placed in 2D ideological space. Each node connects to all others within
radius r (local homophily). Additionally, a fraction of nodes gain random long-range
"bridge" edges that simulate cross-ideological exposure (e.g., social media algorithmic
reach, news aggregators).

Node positions are derived from ideological alignment:
  - x-axis: political spectrum (left=0.0 → right=1.0), jittered within block
  - y-axis: engagement/activity level, sampled uniformly

Block assignment comes from node name prefix (left, center_left, center_right, right).
"""

from typing import Optional, Tuple
import networkx as nx
import numpy as np

# Map name prefix → (block_id, x_center) in [0,1] ideological space
BLOCK_MAP = {
    "left": (0, 0.10),
    "center_left": (1, 0.35),
    "center_right": (2, 0.65),
    "right": (3, 0.90),
}

# RGG connection radius — nodes within this Euclidean distance connect
RGG_RADIUS = 0.30

# Fraction of nodes that receive long-range edges
LONG_RANGE_FRACTION = 0.30

# Number of long-range edges per selected node
LONG_RANGE_K = 2


def _block_for_name(name: str) -> Tuple[int, float]:
    """Return (block_id, x_center) for a node name based on its prefix."""
    for prefix, (block_id, x_center) in BLOCK_MAP.items():
        if name.startswith(prefix):
            return block_id, x_center
    return (1, 0.5)  # fallback: center


def _assign_positions(names: list, seed: Optional[int] = None) -> np.ndarray:
    """
    Place nodes in 2D ideological space.
    x: ideological position (clustered by block, ±0.10 jitter)
    y: engagement level (uniform [0, 1])
    Returns (N, 2) array of positions.
    """
    rng = np.random.default_rng(seed)
    pos = np.zeros((len(names), 2))
    for i, name in enumerate(names):
        _, x_center = _block_for_name(name)
        pos[i, 0] = np.clip(x_center + rng.uniform(-0.10, 0.10), 0.0, 1.0)
        pos[i, 1] = rng.uniform(0.0, 1.0)
    return pos


def create_graph(
    names: list,
    radius: float = RGG_RADIUS,
    long_range_fraction: float = LONG_RANGE_FRACTION,
    long_range_k: int = LONG_RANGE_K,
    seed: Optional[int] = None,
) -> nx.Graph:
    """
    Build a Random Geometric Graph with long-range connections.

    Phase 1 — Local edges (RGG):
        Any two nodes whose Euclidean distance in ideological space <= radius
        are connected. This captures homophily — nearby ideological peers talk.

    Phase 2 — Long-range edges:
        A random subset of nodes (long_range_fraction x N) each gains
        long_range_k random edges to nodes outside their local neighborhood.
        This models cross-ideological exposure and information bridges.

    Node attributes:
        - 'name': persona name string
        - 'block': integer block id (0=left, 1=center_left, 2=center_right, 3=right)
        - 'pos': (x, y) tuple in ideological space
        - 'long_range': True if this node received extra long-range edges

    Edge attributes:
        - 'edge_type': 'local' or 'long_range'
        - 'weight': float (higher for closer local ties, 0.3 for long-range)

    Returns:
        NetworkX Graph with len(names) nodes.
    """
    rng = np.random.default_rng(seed)
    n = len(names)
    pos = _assign_positions(names, seed=seed)

    G = nx.Graph()
    for i, name in enumerate(names):
        block_id, _ = _block_for_name(name)
        G.add_node(i, name=name, block=block_id, pos=tuple(pos[i]), long_range=False)

    # Phase 1: Local RGG edges
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(pos[i] - pos[j])
            if dist <= radius:
                G.add_edge(i, j, edge_type="local", weight=float(1.0 - dist / radius))

    # Phase 2: Long-range edges
    n_long = max(1, int(long_range_fraction * n))
    long_range_nodes = rng.choice(n, size=n_long, replace=False)

    for i in long_range_nodes:
        G.nodes[i]["long_range"] = True
        local_neighbors = set(G.neighbors(i)) | {i}
        candidates = [j for j in range(n) if j not in local_neighbors]
        if not candidates:
            continue
        k = min(long_range_k, len(candidates))
        targets = rng.choice(candidates, size=k, replace=False)
        for j in targets:
            G.add_edge(i, j, edge_type="long_range", weight=0.3)

    return G


def degroot_weights(G: nx.Graph) -> np.ndarray:
    """
    Row-stochastic weight matrix for DeGroot.
    Edge weights are used proportionally; isolated nodes keep their opinion.
    """
    n = G.number_of_nodes()
    W = np.zeros((n, n))
    for i in range(n):
        neighbors = list(G.neighbors(i))
        if not neighbors:
            continue
        weights = np.array([G[i][j].get("weight", 1.0) for j in neighbors])
        weights = weights / weights.sum()
        for j, w in zip(neighbors, weights):
            W[i, j] = w
    return W


def run_degroot(G: nx.Graph, initial_opinions: np.ndarray, steps: int = 5) -> list:
    """
    Run classical DeGroot weighted consensus.

    Args:
        G: Graph with optional 'weight' edge attributes
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


def graph_summary(G: nx.Graph) -> dict:
    """Return key graph statistics for logging."""
    blocks = nx.get_node_attributes(G, "block")
    block_names = ["left", "center_left", "center_right", "right"]
    edge_types = nx.get_edge_attributes(G, "edge_type")
    local_edges = sum(1 for v in edge_types.values() if v == "local")
    long_edges = sum(1 for v in edge_types.values() if v == "long_range")
    long_range_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("long_range"))
    degrees = dict(G.degree())

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "local_edges": local_edges,
        "long_range_edges": long_edges,
        "long_range_nodes": long_range_nodes,
        "avg_degree": sum(degrees.values()) / G.number_of_nodes(),
        "density": nx.density(G),
        "block_sizes": {
            block_names[b]: sum(1 for v in blocks.values() if v == b)
            for b in range(4)
        },
        "is_connected": nx.is_connected(G),
        "components": nx.number_connected_components(G),
    }