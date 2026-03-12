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
import numpy as np

from src.agent import Agent
from src.config import DEFAULT_TOPIC, MAX_CHARS_PER_NEIGHBOR, MAX_NEIGHBORS_PER_UPDATE, PERSONA_BLOCKS
from src.llm_client import get_updated_opinion, prepare_neighbor_opinions
from src.measurement import classify_side_labels, embed_opinions, semantic_variance, opinion_polarization, _get_model


def create_agents(G: nx.Graph, topic: str = DEFAULT_TOPIC) -> List[Agent]:
    """
    Create one agent per node in the ER graph.

    Uses node attributes from the graph (er.py, rgg_long_range.py):
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
    return_side_labels: bool = False,
    persona_set: str = "personas",
) -> tuple[List[float], List[float], List[float], List[dict[str, int]]] | tuple[List[float], List[float], List[float], List[dict[str, int]], list[list[str]]]:
    """
    Run semantic simulation for ``steps`` steps.

    Returns:
        List of semantic variance values (t=0 through t=steps).
    """
    from tqdm import tqdm

    def _counts_from_labels(labels: list[str]) -> dict[str, int]:
        counts = {side: 0 for side in PERSONA_BLOCKS}
        for side in labels:
            if side in counts:
                counts[side] += 1
        return counts

    variances: List[float] = []
    polarizations: List[float] = []
    drifts: List[float] = []
    side_counts: List[dict[str, int]] = []
    side_labels_over_time: list[list[str]] = []

    model = _get_model(show_progress=False)
    real_agents = [a for a in agents if not a.is_bot]
    init_personas = [a.persona_prompt for a in real_agents]
    init_embs = model.encode(init_personas, convert_to_numpy=True, show_progress_bar=False)
    norms_i = np.linalg.norm(init_embs, axis=1, keepdims=True)
    norms_i = np.where(norms_i == 0, 1e-9, norms_i)
    init_embs_norm = init_embs / norms_i

    blocks = [G.nodes[a.node_id].get("block", 0) for a in real_agents]

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
    polarizations.append(opinion_polarization(emb0[:len(real_agents)], blocks))
    
    # Calculate drift
    curr_personas = [a.persona_prompt for a in real_agents]
    curr_embs = model.encode(curr_personas, convert_to_numpy=True, show_progress_bar=False)
    norms_c = np.linalg.norm(curr_embs, axis=1, keepdims=True)
    norms_c = np.where(norms_c == 0, 1e-9, norms_c)
    curr_embs_norm = curr_embs / norms_c
    cos_sim = np.sum(init_embs_norm * curr_embs_norm, axis=1)
    persona_drift = 1.0 - np.clip(cos_sim, -1, 1)
    drifts.append(float(np.mean(persona_drift)))

    labels0 = classify_side_labels(emb0, persona_set=persona_set)
    side_counts.append(_counts_from_labels(labels0))
    if return_side_labels:
        side_labels_over_time.append(labels0)

    step_range = range(1, steps + 1)
    if show_progress:
        step_range = tqdm(step_range, desc="Simulation", unit="step")
    for t in step_range:
        step_t0 = perf_counter()
        step_stats = step_semantic(G, agents, topic)
        opinions = [a.current_opinion for a in agents]
        emb = embed_opinions(opinions)
        variances.append(semantic_variance(emb))
        polarizations.append(opinion_polarization(emb[:len(real_agents)], blocks))

        curr_personas = [a.persona_prompt for a in real_agents]
        curr_embs = model.encode(curr_personas, convert_to_numpy=True, show_progress_bar=False)
        norms_c = np.linalg.norm(curr_embs, axis=1, keepdims=True)
        norms_c = np.where(norms_c == 0, 1e-9, norms_c)
        curr_embs_norm = curr_embs / norms_c
        cos_sim = np.sum(init_embs_norm * curr_embs_norm, axis=1)
        persona_drift = 1.0 - np.clip(cos_sim, -1, 1)
        drifts.append(float(np.mean(persona_drift)))
        
        labels = classify_side_labels(emb, persona_set=persona_set)
        side_counts.append(_counts_from_labels(labels))
        if return_side_labels:
            side_labels_over_time.append(labels)
        llm_updates = int(step_stats["llm_updates"])
        total_llm_updates += llm_updates
        if log_fh is not None:
            record = {
                "type": "step_summary",
                "t": t,
                "semantic_variance": float(variances[-1]),
                "opinion_polarization": float(polarizations[-1]),
                "persona_drift_mean": float(drifts[-1]),
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
    if return_side_labels:
        return variances, polarizations, drifts, side_counts, side_labels_over_time
    return variances, polarizations, drifts, side_counts
