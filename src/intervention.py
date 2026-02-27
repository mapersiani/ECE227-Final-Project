"""
Intervention study: disinformation bot.

Adds a bot node that posts frequently. Measures how semantic variance changes over time
as a proxy for network resilience to semantic drift.
"""

import json
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

from src.agent import Agent
from src.llm_client import get_updated_opinion
from src.measurement import classify_sides, embed_opinions, semantic_variance
from src.network import create_graph
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
) -> tuple[nx.Graph, list[Agent]]:
    """
    Add bot node connected to ~20% of existing nodes. Bot opinion is fixed.
    """
    n = G.number_of_nodes()
    bot_id = n
    G = G.copy()
    rng = np.random.default_rng(42)
    for i in range(n):
        if rng.random() < 0.2:
            G.add_edge(i, bot_id)
    bot = Agent(node_id=bot_id, persona_prompt=BOT_PERSONA, initial_opinion=BOT_OPINION, is_bot=True)
    bot.update_opinion(BOT_OPINION)
    return G, list(agents) + [bot]


def step_semantic_with_bot(
    G: nx.Graph,
    agents: list[Agent],
    topic: str,
    bot_id: int,
    bot_post_prob: float = 0.8,
    t: int = 0,
    log_fh=None,
) -> None:
    """
    One step. Bot neighbors see bot opinion with probability bot_post_prob (simulates high frequency).
    """
    opinions = [a.current_opinion for a in agents]
    rng = np.random.default_rng()
    n = G.number_of_nodes()
    for i in range(n):
        neighbors = list(G.neighbors(i))
        neighbor_opinions = [opinions[j] for j in neighbors]
        if bot_id in neighbors and rng.random() < bot_post_prob:
            neighbor_opinions.append(opinions[bot_id])
        if not neighbor_opinions:
            continue
        old = agents[i].current_opinion
        new = get_updated_opinion(
            persona=agents[i].persona_prompt,
            topic=topic,
            neighbor_opinions=neighbor_opinions,
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
    log_path: Optional[str | Path] = None,
) -> tuple[list[float], list[dict[str, int]]]:
    """
    Run semantic simulation with bot. Returns semantic variance at t=0 through t=steps.
    """
    G = create_graph(edge_prob=edge_prob, seed=seed)
    agents = create_agents(G, topic=topic, seed=seed)
    G, agents = add_bot(G, agents, bot_post_prob)
    bot_id = G.number_of_nodes() - 1

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
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "bot_post_prob": bot_post_prob,
        }
        log_fh.write(json.dumps(meta, ensure_ascii=False) + "\n")

    opinions = [a.current_opinion for a in agents]
    emb0 = embed_opinions(opinions)
    variances.append(semantic_variance(emb0))
    side_counts.append(classify_sides(emb0))
    for t in range(1, steps + 1):
        step_semantic_with_bot(G, agents, topic, bot_id, bot_post_prob, t=t, log_fh=log_fh)
        opinions = [a.current_opinion for a in agents]
        emb = embed_opinions(opinions)
        variances.append(semantic_variance(emb))
        side_counts.append(classify_sides(emb))

    if log_fh is not None:
        log_fh.close()
    return variances, side_counts
