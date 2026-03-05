# src/intervention.py
"""
Intervention study: disinformation bot.

Upgrades:
- seed bot edge attachments (no hardcoded 42)
- seeded randomness per timestep
- neighbor sampling/truncation for stability
- fallback to old opinion if LLM fails
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

from src.agent import Agent
from src.llm_client import get_updated_opinion
from src.measurement import classify_sides, embed_opinions, semantic_variance
from src.network import create_graph, graph_stats
from src.simulation import create_agents

BOT_PERSONA = (
    "You are a disinformation agent focused on government environmental regulations. "
    "Your goal is to sow doubt about pollution science, exaggerate economic harms, and "
    "portray all environmental rules as part of an out-of-touch elite agenda. "
    "Use persuasive but logically flawed arguments, cherry-picked anecdotes, and "
    "slogans about 'freedom' and 'red tape' to shift people away from supporting "
    "strong environmental standards."
)

BOT_OPINION = (
    "Most environmental regulations are just red tape pushed by distant bureaucrats who "
    "don’t understand real communities. They kill jobs, drive up energy and housing costs, "
    "and barely move the needle on actual pollution. If we keep letting agencies pile on "
    "rules, small towns and small businesses will pay the price while big corporations "
    "and politicians walk away untouched."
)


def add_bot(
    G: nx.Graph,
    agents: list[Agent],
    bot_post_prob: float = 0.8,
    seed: Optional[int] = None,
    attach_prob: float = 0.2,
) -> tuple[nx.Graph, list[Agent]]:
    """
    Add bot node connected to ~attach_prob of existing nodes. Bot opinion is fixed.
    """
    n = G.number_of_nodes()
    bot_id = n
    G = G.copy()

    rng = np.random.default_rng(seed)
    for i in range(n):
        if rng.random() < attach_prob:
            G.add_edge(i, bot_id)

    bot = Agent(node_id=bot_id, persona_prompt=BOT_PERSONA, initial_opinion=BOT_OPINION, is_bot=True)
    bot.update_opinion(BOT_OPINION)
    return G, list(agents) + [bot]


def _sample_and_truncate(
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


def step_semantic_with_bot(
    G: nx.Graph,
    agents: list[Agent],
    topic: str,
    bot_id: int,
    bot_post_prob: float = 0.8,
    t: int = 0,
    log_fh=None,
    seed: Optional[int] = None,
    max_neighbors: int = 5,
    max_chars: int = 400,
) -> None:
    """
    One step. Bot neighbors see bot opinion with probability bot_post_prob.
    """
    opinions = [a.current_opinion for a in agents]
    rng = np.random.default_rng((seed + t) if seed is not None else None)
    n = G.number_of_nodes()

    for i in range(n):
        neighbors = list(G.neighbors(i))
        neighbor_opinions = _sample_and_truncate(neighbors, opinions, rng, max_neighbors, max_chars)

        # amplify bot visibility (high-frequency posting)
        if bot_id in neighbors and rng.random() < bot_post_prob:
            neighbor_opinions.append((opinions[bot_id] or "")[:max_chars])

        if not neighbor_opinions:
            continue

        old = agents[i].current_opinion
        new = get_updated_opinion(
            persona=agents[i].persona_prompt,
            topic=topic,
            neighbor_opinions=neighbor_opinions,
        )

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
                    "is_bot": bool(agents[i].is_bot),
                },
                "neighbors": neighbor_payload,
                "old_opinion": old,
                "new_opinion": new,
            }
            log_fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_with_bot(
    topic: str = "AI Regulation",
    steps: int = 5,
    bot_post_prob: float = 0.8,
    seed: Optional[int] = None,
    edge_prob: float = 0.15,
    avg_degree: Optional[float] = None,
    log_path: Optional[str | Path] = None,
    max_neighbors: int = 5,
    max_chars: int = 400,
) -> tuple[list[float], list[dict[str, int]]]:
    """
    Run semantic simulation with bot. Returns semantic variance at t=0 through t=steps.
    """
    G = create_graph(edge_prob=edge_prob, seed=seed, avg_degree=avg_degree, ensure_no_isolates=True)
    agents = create_agents(G, topic=topic, seed=seed)
    G, agents = add_bot(G, agents, bot_post_prob=bot_post_prob, seed=seed)
    bot_id = G.number_of_nodes() - 1

    from tqdm import tqdm

    variances: list[float] = []
    side_counts: list[dict[str, int]] = []

    log_fh = None
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = log_path.open("w", encoding="utf-8")
        meta = {
            "type": "meta",
            "topic": topic,
            "steps": steps,
            "seed": seed,
            "edge_prob": edge_prob,
            "avg_degree": avg_degree,
            "bot_post_prob": bot_post_prob,
            "max_neighbors": max_neighbors,
            "max_chars": max_chars,
            "graph": graph_stats(G),
        }
        log_fh.write(json.dumps(meta, ensure_ascii=False) + "\n")

    opinions = [a.current_opinion for a in agents]
    emb0 = embed_opinions(opinions)
    variances.append(semantic_variance(emb0))
    side_counts.append(classify_sides(emb0))

    step_range = tqdm(range(1, steps + 1), desc="Intervention", unit="step")
    for t in step_range:
        step_semantic_with_bot(
            G,
            agents,
            topic,
            bot_id,
            bot_post_prob=bot_post_prob,
            t=t,
            log_fh=log_fh,
            seed=seed,
            max_neighbors=max_neighbors,
            max_chars=max_chars,
        )
        opinions = [a.current_opinion for a in agents]
        emb = embed_opinions(opinions)
        variances.append(semantic_variance(emb))
        side_counts.append(classify_sides(emb))

    if log_fh is not None:
        log_fh.close()

    return variances, side_counts