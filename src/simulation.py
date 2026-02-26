"""
Discrete-time simulation engine for semantic opinion dynamics.

Each step: agents read neighbors' opinions and query the LLM for an updated opinion.
Tracks semantic variance (embedding-space spread) over time.
"""

from typing import Callable, List, Optional

import networkx as nx
import numpy as np

from src.agent import Agent
from src.config import DEFAULT_TOPIC, PERSONAS
from src.llm_client import get_updated_opinion
from src.measurement import embed_opinions, semantic_variance

_PERSONA_MAP = {p["name"]: p for p in PERSONAS}

def _initial_opinion_for_persona(persona_name: str, topic: str) -> str:
    """Templated initial opinion for persona. Could be replaced with one-shot LLM call."""
    persona = _PERSONA_MAP.get(persona_name)
    if persona and "initial" in persona:
        return persona["initial"]
    # Fallback for any unknown persona
    return f"I have mixed feelings about {topic}."

def create_agents(G: nx.Graph, topic: str = DEFAULT_TOPIC, seed: Optional[int] = None) -> List[Agent]:
    """
    Create one agent per node. Persona comes from node block (0=left … 3=right).
    """
    agents = []
    n = G.number_of_nodes()
    blocks = nx.get_node_attributes(G, "block")
    for i in range(n):
        block = blocks.get(i, 0)
        p = PERSONAS[min(block, len(PERSONAS) - 1)]
        initial = _initial_opinion_for_persona(p["name"], topic)
        agents.append(Agent(node_id=i, persona_prompt=p["prompt"], initial_opinion=initial))
    return agents


def step_semantic(G: nx.Graph, agents: List[Agent], topic: str, memory: str = "") -> None:
    """
    One step: each agent reads neighbors' opinions and calls LLM for updated opinion.
    Updates agents in place.
    """
    opinions = [a.current_opinion for a in agents]
    blocks = nx.get_node_attributes(G, "block")

    for i in range(G.number_of_nodes()):
        neighbors = list(G.neighbors(i))
        neighbor_opinions = [opinions[j] for j in neighbors]
        if not neighbor_opinions:
            continue

        # Look up the style field for this agent's persona
        block = blocks.get(i, 0)
        p = PERSONAS[min(block, len(PERSONAS) - 1)]
        style = p.get("style", "")

        new = get_updated_opinion(
            persona=agents[i].persona_prompt,
            topic=topic,
            neighbor_opinions=neighbor_opinions,
            memory=memory,
            style=style,
        )
        agents[i].update_opinion(new)


def run_semantic(
    G: nx.Graph,
    agents: List[Agent],
    topic: str,
    steps: int = 5,
    on_step: Optional[Callable[[int, List[Agent]], None]] = None,
    show_progress: bool = True,
) -> List[float]:
    """
    Run semantic simulation for `steps` steps.
    Returns:
        List of semantic variance values (t=0 through t=steps).
    """
    from tqdm import tqdm
    variances = []
    opinions = [a.current_opinion for a in agents]
    variances.append(semantic_variance(embed_opinions(opinions)))

    step_range = range(1, steps + 1)
    if show_progress:
        step_range = tqdm(step_range, desc="Simulation", unit="step")

    for t in step_range:
        step_semantic(G, agents, topic)
        opinions = [a.current_opinion for a in agents]
        variances.append(semantic_variance(embed_opinions(opinions)))
        if on_step:
            on_step(t, agents)

    return variances
