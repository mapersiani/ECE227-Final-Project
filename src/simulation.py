# src/simulation.py
"""
Discrete-time simulation engine for semantic opinion dynamics on the ER graph.

Upgrades:
- seeded neighbor sampling (max_neighbors) + truncation (max_chars)
- fallback to old opinion if LLM returns ""
- optional run_seed for reproducibility across runs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, List, Optional

import networkx as nx
import numpy as np

from src.agent import Agent
from src.config import DEFAULT_TOPIC
from src.llm_client import get_updated_opinion
from src.measurement import classify_sides, embed_opinions, semantic_variance
from src.network import graph_stats


def create_agents(G: nx.Graph, topic: str = DEFAULT_TOPIC, seed: Optional[int] = None) -> List[Agent]:
    """
    Create one agent per node in the ER graph.
    Uses node attributes from ``network.create_graph``:
    - prompt
    - initial_text
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


def _sample_and_truncate_neighbors(
    neighbors: list[int],
    opinions: list[str],
    rng: np.random.Generator,
    max_neighbors: int,
    max_chars: int,
) -> list[str]:
    if not neighbors:
        return []
    if max_neighbors is not None and max_neighbors > 0 and len(neighbors) > max_neighbors:
        neighbors = list(rng.choice(neighbors, size=max_neighbors, replace=False))
    out = []
    for j in neighbors:
        txt = opinions[j] if j < len(opinions) else ""
        out.append((txt or "")[: max_chars if max_chars is not None else len(txt)])
    return out


def step_semantic(
    G: nx.Graph,
    agents: List[Agent],
    topic: str,
    t: int,
    memory: str = "",
    log_fh=None,
    seed: Optional[int] = None,
    max_neighbors: int = 5,
    max_chars: int = 400,
) -> None:
    """
    One step: each agent reads (sampled/truncated) neighbors' opinions and calls LLM.
    Updates agents in place.
    """
    opinions = [a.current_opinion for a in agents]
    rng = np.random.default_rng((seed + t) if seed is not None else None)

    for i in range(G.number_of_nodes()):
        neighbors = list(G.neighbors(i))
        neighbor_opinions = _sample_and_truncate_neighbors(neighbors, opinions, rng, max_neighbors, max_chars)
        if not neighbor_opinions:
            continue

        old = agents[i].current_opinion
        new = get_updated_opinion(
            persona=agents[i].persona_prompt,
            topic=topic,
            neighbor_opinions=neighbor_opinions,
            memory=memory,
        )

        # fallback if LLM failed
        if not (new or "").strip():
            new = old

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
    run_seed: Optional[int] = None,
    max_neighbors: int = 5,
    max_chars: int = 400,
) -> tuple[List[float], List[dict[str, int]]]:
    """
    Run semantic simulation for ``steps`` steps.

    Returns:
        (variances, side_counts) for t=0..steps
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
            "run_seed": run_seed,
            "max_neighbors": max_neighbors,
            "max_chars": max_chars,
            "graph": graph_stats(G),
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
        step_semantic(
            G,
            agents,
            topic,
            t=t,
            log_fh=log_fh,
            seed=run_seed,
            max_neighbors=max_neighbors,
            max_chars=max_chars,
        )
        opinions = [a.current_opinion for a in agents]
        emb = embed_opinions(opinions)
        variances.append(semantic_variance(emb))
        side_counts.append(classify_sides(emb))
        if on_step:
            on_step(t, agents)

    if log_fh is not None:
        log_fh.close()

    return variances, side_counts
