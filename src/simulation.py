"""
Discrete-time simulation engine for semantic opinion dynamics on an RGG.

Each step: agents read neighbors' opinions and query the LLM for an updated opinion.
Agents are initialized from nodes.json personas with their actual seed opinions.
Tracks semantic variance (embedding-space spread) over time.
"""

from typing import Callable, List, Optional

import networkx as nx
import numpy as np

from src.agent import Agent
from src.config import DEFAULT_TOPIC, NODES, NODE_NAMES
from src.llm_client import get_updated_opinion
from src.measurement import embed_opinions, semantic_variance


def create_agents(G: nx.Graph, topic: str = DEFAULT_TOPIC, seed: Optional[int] = None) -> List[Agent]:
    """
    Create one agent per node using personas and seed opinions from nodes.json.

    Node i in the graph corresponds to NODE_NAMES[i]. The persona prompt and
    initial opinion are read directly from nodes.json, giving each agent an
    authentic voice rather than a generic block label.

    Args:
        G: Graph built from create_graph(NODE_NAMES, ...)
        topic: Simulation topic (currently informational; initial opinions are from nodes.json)
        seed: Unused (kept for API compatibility)

    Returns:
        List of Agent objects, one per node.
    """
    agents = []
    n = G.number_of_nodes()
    node_names_in_graph = nx.get_node_attributes(G, "name")

    for i in range(n):
        name = node_names_in_graph.get(i, NODE_NAMES[i] if i < len(NODE_NAMES) else "unknown")
        # Find matching node data
        node_data = next((nd for nd in NODES if nd["name"] == name), None)
        if node_data is None:
            # Fallback if name not found
            persona_prompt = f"You are a thoughtful citizen with views on {topic}."
            initial_opinion = f"I have mixed feelings about {topic}."
        else:
            persona_prompt = node_data["prompt"]
            initial_opinion = node_data["initial"]

        agents.append(Agent(
            node_id=i,
            persona_prompt=persona_prompt,
            initial_opinion=initial_opinion,
        ))
    return agents


def step_semantic(G: nx.Graph, agents: List[Agent], topic: str, memory: str = "") -> None:
    """
    One simulation step: each agent reads neighbors' opinions and calls LLM for update.
    Updates agents in place (synchronous snapshot → update).
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