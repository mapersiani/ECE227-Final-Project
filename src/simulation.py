"""Discrete-time simulation engine for semantic opinion dynamics."""

from typing import Callable, List, Optional

import networkx as nx
import numpy as np

from src.agent import Agent
from src.config import DEFAULT_TOPIC, PERSONAS
from src.llm_client import get_updated_opinion
from src.measurement import embed_opinions, semantic_variance


def _initial_opinion_for_persona(persona_name: str, topic: str) -> str:
    """Generate a placeholder initial opinion; in practice, could call LLM once."""
    templates = {
        "left": f"On {topic}, I support strong regulation to protect the public and curb corporate overreach.",
        "center_left": f"On {topic}, I favor regulation that enables innovation while safeguarding society.",
        "center_right": f"On {topic}, I prefer light-touch regulation that lets markets lead with targeted oversight.",
        "right": f"On {topic}, I oppose heavy regulation; voluntary standards and markets are sufficient.",
    }
    return templates.get(persona_name, f"I have mixed feelings about {topic}.")


def create_agents(
    G: nx.Graph,
    topic: str = DEFAULT_TOPIC,
    seed: Optional[int] = None,
) -> list[Agent]:
    """Create agents with personas from node block (SBM: block 0=left, 1=center_left, etc.)."""
    agents = []
    n = G.number_of_nodes()
    blocks = nx.get_node_attributes(G, "block")
    for i in range(n):
        block = blocks.get(i, 0)
        p = PERSONAS[min(block, len(PERSONAS) - 1)]
        initial = _initial_opinion_for_persona(p["name"], topic)
        agents.append(
            Agent(
                node_id=i,
                persona_prompt=p["prompt"],
                initial_opinion=initial,
            )
        )
    return agents


def step_semantic(
    G: nx.Graph,
    agents: list[Agent],
    topic: str,
    memory: str = "",
) -> None:
    """
    One discrete-time step: each agent reads neighbors' opinions and updates via LLM.
    """
    opinions = [a.current_opinion for a in agents]
    for i in range(G.number_of_nodes()):
        neighbors = list(G.neighbors(i))
        neighbor_opinions = [opinions[j] for j in neighbors]
        if not neighbor_opinions:
            continue
        new_opinion = get_updated_opinion(
            persona=agents[i].persona_prompt,
            topic=topic,
            neighbor_opinions=neighbor_opinions,
            memory=memory,
        )
        agents[i].update_opinion(new_opinion)


def run_semantic(
    G: nx.Graph,
    agents: list[Agent],
    topic: str,
    steps: int = 5,
    on_step: Optional[Callable[[int, list[Agent]], None]] = None,
) -> list[float]:
    """
    Run semantic simulation for `steps` timesteps.

    Returns:
        List of semantic variance values (one per timestep, including t=0).
    """
    variances = []
    opinions = [a.current_opinion for a in agents]
    emb = embed_opinions(opinions)
    variances.append(semantic_variance(emb))

    for t in range(1, steps + 1):
        step_semantic(G, agents, topic)
        opinions = [a.current_opinion for a in agents]
        emb = embed_opinions(opinions)
        variances.append(semantic_variance(emb))
        if on_step:
            on_step(t, agents)

    return variances

