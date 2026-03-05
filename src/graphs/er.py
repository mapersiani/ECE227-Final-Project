"""ER graph adapters used by experiment runners."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import networkx as nx

from src.network import create_graph, export_gephi


def create_er_graph(edge_prob: float = 0.15, seed: Optional[int] = None) -> nx.Graph:
    """Create an Erdős–Rényi graph over personas from data/nodes.json."""
    return create_graph(edge_prob=edge_prob, seed=seed)


def export_er_graph(G: nx.Graph, out_dir: Path, basename: str) -> tuple[Path, Path]:
    """Export ER graph into Gephi-friendly formats."""
    return export_gephi(G, out_dir, basename)
