"""ER graph adapters used by experiment runners."""

from __future__ import annotations

from typing import Optional

import networkx as nx

from src.network import create_graph


def create_er_graph(edge_prob: float = 0.15, seed: Optional[int] = None) -> nx.Graph:
    """Create an Erdős–Rényi graph over personas from data/nodes.json."""
    return create_graph(edge_prob=edge_prob, seed=seed)
