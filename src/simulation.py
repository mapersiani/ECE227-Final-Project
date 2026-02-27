"""
Discrete-time simulation engine for semantic opinion dynamics on the ER graph.

Each step: agents read neighbors' opinions and query the LLM for an updated opinion.
Tracks semantic variance (embedding-space spread) over time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional

import networkx as nx

from src.agent import Agent
from src.config import DEFAULT_TOPIC
from src.llm_client import get_updated_opinion
from src.measurement import classify_sides, embed_opinions, semantic_variance


def create_agents(G: nx.Graph, topic: str = DEFAULT_TOPIC, seed: Optional[int] = None) -> List[Agent]:
    """
    Create one agent per node in the ER graph.

    Uses node attributes from ``network.create_graph``:
    - ``prompt``: persona prompt for the LLM
    - ``initial_text``: initial opinion text
    """
    agents: List[Agent] = []
    for node_id, data in G.nodes(data=True):
        persona_prompt = data.get("prompt", "")
        initial_text = data.get("initial_text") or f"I have mixed feelings about {topic}."
        agents.append(
            Agent(
                node_id=node_id,
                persona_prompt=persona_prompt,
                initial_opinion=initial_text,
            )
        )
    return agents


def step_semantic(
    G: nx.Graph,
    agents: List[Agent],
    topic: str,
    t: int,
    memory: str = "",
    log_fh=None,
) -> None:
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
        old = agents[i].current_opinion
        new = get_updated_opinion(
            persona=agents[i].persona_prompt,
            topic=topic,
            neighbor_opinions=neighbor_opinions,
            memory=memory,
        )
        agents[i].update_opinion(new)
        if log_fh is not None:
            neighbor_payload = []
            for j in neighbors:
                nd = G.nodes[j]
                neighbor_payload.append(
                    {
                        "node_id": j,
                        "name": nd.get("name", str(j)),
                        "side": nd.get("side", "unknown"),
                        "opinion": opinions[j],
                    }
                )
            nd_i = G.nodes[i]
            record = {
                "type": "interaction",
                "t": t,
                "topic": topic,
                "node": {
                    "node_id": i,
                    "name": nd_i.get("name", str(i)),
                    "side": nd_i.get("side", "unknown"),
                },
                "neighbors": neighbor_payload,
                "old_opinion": old,
                "new_opinion": new,
            }
            log_fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_semantic(
    G: nx.Graph,
    agents: List[Agent],
    topic: str,
    steps: int = 5,
    on_step: Optional[Callable[[int, List[Agent]], None]] = None,
    show_progress: bool = True,
    log_path: Optional[str | Path] = None,
) -> tuple[List[float], List[dict[str, int]]]:
    """
    Run semantic simulation for ``steps`` steps.

    Returns:
        List of semantic variance values (t=0 through t=steps).
    """
    from tqdm import tqdm

    variances: List[float] = []
    side_counts: List[dict[str, int]] = []

    log_fh = None
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = log_path.open("w", encoding="utf-8")
        meta = {
            "type": "meta",
            "topic": topic,
            "steps": steps,
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        }
        log_fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
    opinions = [a.current_opinion for a in agents]
    emb0 = embed_opinions(opinions)
    variances.append(semantic_variance(emb0))
    side_counts.append(classify_sides(emb0))

    step_range = range(1, steps + 1)
    if show_progress:
        step_range = tqdm(step_range, desc="Simulation", unit="step")
    for t in step_range:
        step_semantic(G, agents, topic, t=t, log_fh=log_fh)
        opinions = [a.current_opinion for a in agents]
        emb = embed_opinions(opinions)
        variances.append(semantic_variance(emb))
        side_counts.append(classify_sides(emb))
        if on_step:
            on_step(t, agents)
    if log_fh is not None:
        log_fh.close()
    return variances, side_counts
