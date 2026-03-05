"""
Black-box LLM simulation with implicit (LLM-internal) utility and update rule.

No explicit payoff matrix or explicit strategy-update equation is used in the loop.
SBERT is only used at the end for performance evaluation.
"""

from typing import Optional

import networkx as nx
import numpy as np
from src.llm_client import get_updated_opinion
from src.measurement import embed_opinions, semantic_variance
from src.simulation import create_agents


def run_hybrid_llm_egt(
    G: nx.Graph,
    topic: str,
    steps: int = 5,
    seed: Optional[int] = None,
) -> dict:
    """
    Run fully black-box LLM updates.

    Returns:
      - opinions_history: list[list[str]]
      - initial_semantic_variance: float
      - final_semantic_variance: float
      - semantic_drift: float (mean L2 distance between initial/final embeddings)
    """
    agents = create_agents(G, topic=topic, seed=seed)
    n = G.number_of_nodes()

    opinions = [a.current_opinion for a in agents]
    opinions_history = [opinions.copy()]
    initial_embeddings = embed_opinions(opinions)
    initial_semantic_variance = semantic_variance(initial_embeddings)

    for _ in range(steps):
        frozen_opinions = [a.current_opinion for a in agents]

        for i in range(n):
            neighbors = list(G.neighbors(i))
            neighbor_opinions = [frozen_opinions[j] for j in neighbors]
            if not neighbor_opinions:
                continue

            memory = (
                "Decide your own update rule implicitly. Internally weigh persuasive strength, "
                "social agreement, and long-term self-interest using your persona as the anchor. "
                "You may keep, soften, or shift your stance, but avoid abrupt contradiction unless "
                "neighbor evidence is genuinely strong."
            )
            new_opinion = get_updated_opinion(
                persona=agents[i].persona_prompt,
                topic=topic,
                neighbor_opinions=neighbor_opinions,
                memory=memory,
            )
            agents[i].update_opinion(new_opinion)

        opinions = [a.current_opinion for a in agents]
        opinions_history.append(opinions.copy())

    final_embeddings = embed_opinions(opinions)
    final_semantic_variance = semantic_variance(final_embeddings)
    semantic_drift = float(np.mean(np.linalg.norm(final_embeddings - initial_embeddings, axis=1)))

    return {
        "opinions_history": opinions_history,
        "initial_semantic_variance": initial_semantic_variance,
        "final_semantic_variance": final_semantic_variance,
        "semantic_drift": semantic_drift,
    }
