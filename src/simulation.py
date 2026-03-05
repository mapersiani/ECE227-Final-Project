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


def _initial_opinion_for_persona(persona_name: str, topic: str) -> str:
    """Templated initial opinion for persona. Could be replaced with one-shot LLM call."""
    templates = {
        "left": f"On {topic}, I support strong regulation to protect the public and curb corporate overreach.",
        "center_left": f"On {topic}, I favor regulation that enables innovation while safeguarding society.",
        "center_right": f"On {topic}, I prefer light-touch regulation that lets markets lead with targeted oversight.",
        "right": f"On {topic}, I oppose heavy regulation; voluntary standards and markets are sufficient.",
    }
    return templates.get(persona_name, f"I have mixed feelings about {topic}.")


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


def create_agents_from_nodes_data(
    G: nx.Graph,
    nodes_data: List[dict],
    topic: str = DEFAULT_TOPIC,
) -> List[Agent]:
    """
    Create agents from external node records (e.g., Node.js-generated JSON).

    Expected fields per node:
      - prompt: persona/system prompt
      - initial: initial opinion text
    """
    agents = []
    n = G.number_of_nodes()
    if len(nodes_data) < n:
        raise ValueError(f"nodes_data has {len(nodes_data)} entries but graph has {n} nodes")

    for i in range(n):
        rec = nodes_data[i]
        persona = rec.get("prompt", "You are a neutral participant.")
        initial = rec.get("initial", f"I have mixed feelings about {topic}.")
        agents.append(Agent(node_id=i, persona_prompt=persona, initial_opinion=initial))
    return agents


def step_semantic(G: nx.Graph, agents: List[Agent], topic: str, memory: str = "") -> None:
    """
    One step: each agent reads neighbors' opinions and calls LLM for updated opinion.
    Updates agents in place.
    """
    opinions = [a.current_opinion for a in agents]
    for i in range(G.number_of_nodes()):
        neighbors = list(G.neighbors(i))
        neighbor_opinions = [opinions[j] for j in neighbors]
        if not neighbor_opinions:
            continue
        new = get_updated_opinion(
            persona=agents[i].persona_prompt,
            topic=topic,
            neighbor_opinions=neighbor_opinions,
            memory=memory,
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
