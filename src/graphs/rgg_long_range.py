"""Random geometric graph with long-range ties."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import networkx as nx
import numpy as np

from src.config import (
    DEFAULT_SEED,
    LONG_RANGE_FRACTION,
    LONG_RANGE_K,
    PERSONA_BLOCK_LAYOUT,
    PERSONA_BLOCKS,
    RGG_RADIUS,
    side_from_name,
)


@dataclass(frozen=True)
class RGGLongRangeParams:
    """Parameters for random geometric graphs with long-range edges."""

    radius: float = RGG_RADIUS
    long_range_fraction: float = LONG_RANGE_FRACTION
    long_range_k: int = LONG_RANGE_K
    seed: Optional[int] = DEFAULT_SEED


def _side_for_name(name: str) -> str:
    return side_from_name(name)


def _block_for_name(name: str) -> Tuple[int, float]:
    side = _side_for_name(name)
    if side in PERSONA_BLOCK_LAYOUT:
        return PERSONA_BLOCK_LAYOUT[side]
    return (1, 0.50)


def _assign_positions(names: list[str], seed: Optional[int]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = np.zeros((len(names), 2))
    for i, name in enumerate(names):
        _, x_center = _block_for_name(name)
        pos[i, 0] = np.clip(x_center + rng.uniform(-0.10, 0.10), 0.0, 1.0)
        pos[i, 1] = rng.uniform(0.0, 1.0)
    return pos


def create_rgg_long_range_graph(nodes: list[dict], params: RGGLongRangeParams) -> nx.Graph:
    """Build a graph from persona records with local and long-range edges. Node count is dynamic."""
    rng = np.random.default_rng(params.seed)
    names = [n.get("name", f"node_{i}") for i, n in enumerate(nodes)]
    n = len(names)
    pos = _assign_positions(names, seed=params.seed)

    G = nx.Graph()
    for i, rec in enumerate(nodes):
        name = names[i]
        block, _ = _block_for_name(name)
        side = _side_for_name(name)
        G.add_node(
            i,
            name=name,
            prompt=rec.get("prompt", ""),
            style=rec.get("style", ""),
            initial_text=rec.get("initial", ""),
            side=side,
            block=block,
            pos=tuple(pos[i]),
            long_range=False,
        )

    # Local geometric edges.
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(pos[i] - pos[j]))
            if dist <= params.radius:
                weight = float(max(0.0, 1.0 - dist / params.radius))
                G.add_edge(i, j, edge_type="local", weight=weight)

    # Long-range bridge edges.
    n_long = max(1, int(params.long_range_fraction * n))
    long_nodes = rng.choice(n, size=n_long, replace=False)
    for i in long_nodes:
        G.nodes[i]["long_range"] = True
        local_neighbors = set(G.neighbors(i)) | {i}
        candidates = [j for j in range(n) if j not in local_neighbors]
        if not candidates:
            continue
        k = min(params.long_range_k, len(candidates))
        targets = rng.choice(candidates, size=k, replace=False)
        for j in targets:
            G.add_edge(i, int(j), edge_type="long_range", weight=0.3)

    return G
