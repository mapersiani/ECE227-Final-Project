"""Semantic simulation adapters for integration branch."""

from __future__ import annotations

from typing import Optional

import networkx as nx

from src.agent import Agent
from src.simulation import create_agents, run_semantic


def create_agents_for_graph(G: nx.Graph, topic: str, seed: Optional[int] = None) -> list[Agent]:
    """Compatibility wrapper for agent creation."""
    return create_agents(G, topic=topic, seed=seed)


def run_semantic_simulation(
    G: nx.Graph,
    agents: list[Agent],
    topic: str,
    steps: int,
    log_path: Optional[str] = None,
) -> tuple[list[float], list[dict[str, int]]]:
    """Compatibility wrapper for semantic simulation."""
    return run_semantic(G, agents, topic, steps=steps, log_path=log_path)
