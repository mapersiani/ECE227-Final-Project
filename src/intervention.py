"""
Intervention study: disinformation bot.

Adds a bot node that posts frequently. Measures how semantic variance changes over time
as a proxy for network resilience to semantic drift.
"""

import json
from pathlib import Path
from time import perf_counter
from typing import Optional

import networkx as nx
import numpy as np

from src.agent import Agent
from src.config import (
    BOT_DEPLOY_STEPS,
    BOT_INJECTION_STEP,
    DEFAULT_STEPS,
    MAX_CHARS_PER_NEIGHBOR,
    MAX_NEIGHBORS_PER_UPDATE,
    PERSONA_BLOCKS,
)
from src.llm_client import get_updated_opinion, prepare_neighbor_opinions
from src.measurement import classify_side_labels, embed_opinions, semantic_variance
from src.simulation import create_agents

DEFAULT_BOT_PERSONA = (
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
BOT_PROFILE_PATH = Path(__file__).resolve().parent.parent / "data" / "bot_profiles" / "trump_env_reg.json"
_BOT_PROFILE_CACHE: dict[str, str] | None = None


def _load_bot_profile() -> dict[str, str]:
    global _BOT_PROFILE_CACHE
    if _BOT_PROFILE_CACHE is not None:
        return _BOT_PROFILE_CACHE

    profile = {
        "name": "bot_disinfo",
        "persona_prompt": DEFAULT_BOT_PERSONA,
        "initial_opinion": BOT_OPINION,
    }
    if BOT_PROFILE_PATH.exists():
        with BOT_PROFILE_PATH.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        profile["name"] = str(raw.get("name", profile["name"]))
        profile["persona_prompt"] = str(raw.get("prompt", profile["persona_prompt"]))
        # Prefer nodes-style key "initial", fallback to legacy "initial_opinion".
        profile["initial_opinion"] = str(
            raw.get("initial", raw.get("initial_opinion", profile["initial_opinion"]))
        )

    _BOT_PROFILE_CACHE = profile
    return profile


def add_bot(
    G: nx.Graph,
    agents: list[Agent],
    seed: Optional[int] = None,
) -> tuple[nx.Graph, list[Agent]]:
    """
    Add bot node connected to ~20% of existing nodes. Bot opinion is fixed.
    """
    profile = _load_bot_profile()
    bot_name = profile["name"]
    bot_persona = profile["persona_prompt"]
    bot_opinion = profile["initial_opinion"]

    n = G.number_of_nodes()
    bot_id = n
    G = G.copy()
    G.add_node(
        bot_id,
        name=bot_name,
        side="bot",
        prompt=bot_persona,
        initial_text=bot_opinion,
        is_bot=True,
    )
    rng = np.random.default_rng(seed)
    for i in range(n):
        if rng.random() < 0.2:
            G.add_edge(i, bot_id)
    bot = Agent(node_id=bot_id, persona_prompt=bot_persona, initial_opinion=bot_opinion, is_bot=True)
    bot.update_opinion(bot_opinion)
    return G, list(agents) + [bot]


def step_semantic_with_bot(
    G: nx.Graph,
    agents: list[Agent],
    topic: str,
    bot_id: int,
    t: int,
    bot_post_prob: float = 0.8,
) -> dict[str, float]:
    """
    One step. Bot message is deployed only on configured steps.
    """
    opinions = [a.current_opinion for a in agents]
    bot_deployed = t in BOT_DEPLOY_STEPS
    n = G.number_of_nodes()
    llm_updates = 0
    total_neighbors_used = 0
    total_neighbor_chars = 0
    bot_amplified_updates = 0
    for i in range(n):
        if i == bot_id:
            continue
        neighbors = list(G.neighbors(i))
        neighbor_opinions = [opinions[j] for j in neighbors]
        bot_amplified = False
        if bot_deployed and bot_id in neighbors:
            neighbor_opinions.append(opinions[bot_id])
            bot_amplified = True
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
            opinions_prepared=True,
        )
        agents[i].update_opinion(new)
        llm_updates += 1
        if bot_amplified:
            bot_amplified_updates += 1

    avg_neighbors_used = (total_neighbors_used / llm_updates) if llm_updates else 0.0
    avg_neighbor_chars = (total_neighbor_chars / llm_updates) if llm_updates else 0.0
    return {
        "llm_updates": float(llm_updates),
        "avg_neighbors_used": float(avg_neighbors_used),
        "avg_neighbor_chars": float(avg_neighbor_chars),
        "bot_amplified_updates": float(bot_amplified_updates),
        "bot_deployed": 1.0 if bot_deployed else 0.0,
    }


def run_with_bot_on_graph(
    G: nx.Graph,
    topic: str,
    steps: int = DEFAULT_STEPS,
    bot_post_prob: float = 0.8,
    seed: Optional[int] = None,
    log_path: Optional[str | Path] = None,
    show_progress: bool = True,
    return_state: bool = False,
    return_side_labels: bool = False,
    persona_set: str = "personas",
) -> (
    tuple[list[float], list[dict[str, int]]]
    | tuple[list[float], list[dict[str, int]], list[list[str]]]
    | tuple[list[float], list[dict[str, int]], nx.Graph, list[Agent]]
    | tuple[list[float], list[dict[str, int]], list[list[str]], nx.Graph, list[Agent]]
):
    """Run semantic simulation with bot on a provided graph (bot injected at t=0)."""
    agents = create_agents(G, topic=topic)
    G, agents = add_bot(G, agents, seed=seed)
    bot_id = G.number_of_nodes() - 1

    from tqdm import tqdm

    variances: list[float] = []
    side_counts: list[dict[str, int]] = []
    side_labels_over_time: list[list[str]] = []

    def _counts_from_labels(labels: list[str]) -> dict[str, int]:
        counts = {side: 0 for side in PERSONA_BLOCKS}
        for side in labels:
            if side in counts:
                counts[side] += 1
        return counts

    log_fh = None
    run_t0 = perf_counter()
    total_llm_updates = 0
    total_bot_amplified_updates = 0
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
            "bot_deploy_steps": list(BOT_DEPLOY_STEPS),
            "bot_injection_step": BOT_INJECTION_STEP,
            "log_schema": "summary_v1",
            "max_neighbors_per_update": MAX_NEIGHBORS_PER_UPDATE,
            "max_chars_per_neighbor": MAX_CHARS_PER_NEIGHBOR,
        }
        log_fh.write(json.dumps(meta, ensure_ascii=False) + "\n")

    opinions = [a.current_opinion for a in agents]
    emb0 = embed_opinions(opinions)
    variances.append(semantic_variance(emb0))
    labels0 = classify_side_labels(emb0, persona_set=persona_set)
    side_counts.append(_counts_from_labels(labels0))
    if return_side_labels:
        side_labels_over_time.append(labels0)

    step_range = range(1, steps + 1)
    if show_progress:
        step_range = tqdm(step_range, desc="Intervention", unit="step")
    for t in step_range:
        step_t0 = perf_counter()
        step_stats = step_semantic_with_bot(G, agents, topic, bot_id, t, bot_post_prob)
        opinions = [a.current_opinion for a in agents]
        emb = embed_opinions(opinions)
        variances.append(semantic_variance(emb))
        labels = classify_side_labels(emb, persona_set=persona_set)
        side_counts.append(_counts_from_labels(labels))
        if return_side_labels:
            side_labels_over_time.append(labels)
        llm_updates = int(step_stats["llm_updates"])
        bot_amplified_updates = int(step_stats["bot_amplified_updates"])
        total_llm_updates += llm_updates
        total_bot_amplified_updates += bot_amplified_updates
        if log_fh is not None:
            record = {
                "type": "step_summary",
                "t": t,
                "semantic_variance": float(variances[-1]),
                "side_counts": side_counts[-1],
                "llm_updates": llm_updates,
                "avg_neighbors_used": float(step_stats["avg_neighbors_used"]),
                "avg_neighbor_chars": float(step_stats["avg_neighbor_chars"]),
                "bot_amplified_updates": bot_amplified_updates,
                "bot_deployed": bool(step_stats["bot_deployed"]),
                "step_elapsed_sec": round(perf_counter() - step_t0, 4),
            }
            log_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    if log_fh is not None:
        final = {
            "type": "run_summary",
            "total_llm_updates": int(total_llm_updates),
            "total_bot_amplified_updates": int(total_bot_amplified_updates),
            "total_elapsed_sec": round(perf_counter() - run_t0, 4),
        }
        log_fh.write(json.dumps(final, ensure_ascii=False) + "\n")
        log_fh.close()
    if return_state:
        if return_side_labels:
            return variances, side_counts, side_labels_over_time, G, agents
        return variances, side_counts, G, agents
    if return_side_labels:
        return variances, side_counts, side_labels_over_time
    return variances, side_counts
