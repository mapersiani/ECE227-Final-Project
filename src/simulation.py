"""
Discrete-time simulation engine for semantic opinion dynamics on the ER graph.

Each step: agents read neighbors' opinions and query the LLM for an updated opinion.
Tracks semantic variance (embedding-space spread) over time.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Callable, List, Optional

import networkx as nx

from src.agent import Agent
from src.config import DEFAULT_TOPIC, MAX_CHARS_PER_NEIGHBOR, MAX_NEIGHBORS_PER_UPDATE
from src.llm_client import get_updated_opinion, prepare_neighbor_opinions
from src.measurement import classify_sides, embed_opinions, semantic_variance


def create_agents(G: nx.Graph, topic: str = DEFAULT_TOPIC) -> List[Agent]:
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
    memory: str = "",
) -> dict[str, float]:
    """
    One step: each agent reads neighbors' opinions and calls LLM for updated opinion.
    Updates agents in place.
    """
    opinions = [a.current_opinion for a in agents]
    llm_updates = 0
    total_neighbors_used = 0
    total_neighbor_chars = 0
    for i in range(G.number_of_nodes()):
        neighbors = list(G.neighbors(i))
        neighbor_opinions = [opinions[j] for j in neighbors]
        prepared_neighbors = prepare_neighbor_opinions(
            neighbor_opinions,
            max_neighbors=MAX_NEIGHBORS_PER_UPDATE,
            max_chars_per_neighbor=MAX_CHARS_PER_NEIGHBOR,
        )
        total_neighbors_used += len(prepared_neighbors)
        total_neighbor_chars += sum(len(txt) for txt in prepared_neighbors)
        if not prepared_neighbors:
            continue
        new = get_updated_opinion(
            persona=agents[i].persona_prompt,
            topic=topic,
            neighbor_opinions=prepared_neighbors,
            memory=memory,
            opinions_prepared=True,
        )
        agents[i].update_opinion(new)
        llm_updates += 1

    avg_neighbors_used = (total_neighbors_used / llm_updates) if llm_updates else 0.0
    avg_neighbor_chars = (total_neighbor_chars / llm_updates) if llm_updates else 0.0
    return {
        "llm_updates": float(llm_updates),
        "avg_neighbors_used": float(avg_neighbors_used),
        "avg_neighbor_chars": float(avg_neighbor_chars),
    }


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
    run_t0 = perf_counter()
    total_llm_updates = 0
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
            "log_schema": "summary_v1",
            "max_neighbors_per_update": MAX_NEIGHBORS_PER_UPDATE,
            "max_chars_per_neighbor": MAX_CHARS_PER_NEIGHBOR,
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
        step_t0 = perf_counter()
        step_stats = step_semantic(G, agents, topic)
        opinions = [a.current_opinion for a in agents]
        emb = embed_opinions(opinions)
        variances.append(semantic_variance(emb))
        side_counts.append(classify_sides(emb))
        llm_updates = int(step_stats["llm_updates"])
        total_llm_updates += llm_updates
        if log_fh is not None:
            record = {
                "type": "step_summary",
                "t": t,
                "semantic_variance": float(variances[-1]),
                "side_counts": side_counts[-1],
                "llm_updates": llm_updates,
                "avg_neighbors_used": float(step_stats["avg_neighbors_used"]),
                "avg_neighbor_chars": float(step_stats["avg_neighbor_chars"]),
                "step_elapsed_sec": round(perf_counter() - step_t0, 4),
            }
            log_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        if on_step:
            on_step(t, agents)
    if log_fh is not None:
        final = {
            "type": "run_summary",
            "total_llm_updates": int(total_llm_updates),
            "total_elapsed_sec": round(perf_counter() - run_t0, 4),
        }
        log_fh.write(json.dumps(final, ensure_ascii=False) + "\n")
        log_fh.close()
    return variances, side_counts
