"""Intervention study: disinformation bot with high posting frequency."""

from typing import List, Optional

import networkx as nx
import numpy as np

from src.agent import Agent
from src.llm_client import get_updated_opinion
from src.measurement import embed_opinions, semantic_variance
from src.network import create_graph
from src.simulation import create_agents


BOT_PERSONA = (
    "You are a disinformation agent. Your goal is to spread misleading or inflammatory "
    "claims to destabilize consensus. Use persuasive but flawed reasoning."
)
BOT_OPINION = (
    "AI Regulation is a government overreach that will stifle innovation and give "
    "unelected bureaucrats control over technology. The real danger is not AI, but those "
    "who want to regulate it."
)


def add_bot(
    G: nx.Graph,
    agents: List[Agent],
    topic: str,
    bot_post_prob: float = 0.8,
) -> tuple[nx.Graph, List[Agent]]:
    """
    Add a disinformation bot node with high posting frequency.
    Bot connects to ~20% of existing nodes.
    """
    n = G.number_of_nodes()
    bot_id = n
    G = G.copy()
    rng = np.random.default_rng(42)
    for i in range(n):
        if rng.random() < 0.2:
            G.add_edge(i, bot_id)
    bot = Agent(
        node_id=bot_id,
        persona_prompt=BOT_PERSONA,
        initial_opinion=BOT_OPINION,
        is_bot=True,
    )
    bot.update_opinion(BOT_OPINION)
    agents = list(agents) + [bot]
    return G, agents


def step_semantic_with_bot(
    G: nx.Graph,
    agents: List[Agent],
    topic: str,
    bot_id: int,
    bot_post_prob: float = 0.8,
) -> None:
    """One step where the bot's opinion is injected with probability bot_post_prob."""
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
        new_opinion = get_updated_opinion(
            persona=agents[i].persona_prompt,
            topic=topic,
            neighbor_opinions=neighbor_opinions,
        )
        agents[i].update_opinion(new_opinion)


def run_with_bot(
    topic: str = "AI Regulation",
    steps: int = 5,
    bot_post_prob: float = 0.8,
    seed: Optional[int] = None,
) -> List[float]:
    """Run simulation with disinformation bot on SBM graph."""
    G = create_graph(seed=seed)
    agents = create_agents(G, topic=topic, seed=seed)
    G, agents = add_bot(G, agents, topic, bot_post_prob)
    bot_id = G.number_of_nodes() - 1

    variances = []
    opinions = [a.current_opinion for a in agents]
    emb = embed_opinions(opinions)
    variances.append(semantic_variance(emb))

    for _ in range(steps):
        step_semantic_with_bot(G, agents, topic, bot_id, bot_post_prob)
        opinions = [a.current_opinion for a in agents]
        emb = embed_opinions(opinions)
        variances.append(semantic_variance(emb))

    return variances
