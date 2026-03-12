"""Erdős–Rényi graph builder for opinion dynamics experiments."""

from __future__ import annotations

from typing import Optional

import networkx as nx

from src.config import side_from_name
from src.load_nodes import load_nodes


def create_er_graph(
    edge_prob: float = 0.15,
    seed: Optional[int] = None,
    persona_set: str = "personas",
) -> nx.Graph:
    """
    Create an Erdős–Rényi graph on personas from the selected nodes file.

    Args:
        edge_prob: ER edge probability p (0–1)
        seed: Random seed for reproducibility
        persona_set: "personas" (data/nodes.json) or "senate" (data/senate_nodes.json)

    Returns:
        NetworkX Graph with one node per persona and attributes:
        - name, prompt, style, initial_text, side (democrat/republican/independent)
    """
    nodes = load_nodes(persona_set)
    n = len(nodes)
    G = nx.erdos_renyi_graph(n=n, p=edge_prob, seed=seed)

    attrs: dict[int, dict] = {}
    for i, node in enumerate(nodes):
        name = node.get("name", f"node_{i}")
        side = side_from_name(name)
        attrs[i] = {
            "name": name,
            "prompt": node.get("prompt", ""),
            "style": node.get("style", ""),
            "initial_text": node.get("initial", ""),
            "side": side,
        }
    nx.set_node_attributes(G, attrs)
    return G
