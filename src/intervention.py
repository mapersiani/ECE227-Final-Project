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
    DEFAULT_STEPS,
    MAX_CHARS_PER_NEIGHBOR,
    MAX_NEIGHBORS_PER_UPDATE,
    PERSONA_BLOCKS,
)
from src.llm_client import get_updated_opinion, prepare_neighbor_opinions, get_vote
from src.measurement import (
    classify_side_labels,
    classify_sides,
    embed_opinions,
    mean_persona_drift,
    opinion_polarization,
    semantic_variance,
    _get_model,
)
from src.simulation import create_agents

BOT_PROFILE_PATH = Path(__file__).resolve().parent.parent / "data" / "bot_profiles" / "trump_env_reg.json"
_BOT_PROFILE_CACHE: dict[str, str] | None = None


def _load_bot_profile() -> dict[str, str]:
    """Load bot profile from trump_env_reg.json. Raises FileNotFoundError if missing."""
    global _BOT_PROFILE_CACHE
    if _BOT_PROFILE_CACHE is not None:
        return _BOT_PROFILE_CACHE

    if not BOT_PROFILE_PATH.exists():
        raise FileNotFoundError(
            f"Bot profile not found: {BOT_PROFILE_PATH}\n"
            "The bot requires data/bot_profiles/trump_env_reg.json to exist."
        )

    with BOT_PROFILE_PATH.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    profile = {
        "name": str(raw.get("name", "bot_disinfo")),
        "persona_prompt": str(raw["prompt"]),
        "initial_opinion": str(raw.get("initial", raw.get("initial_opinion", ""))),
    }

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
            current_opinion=agents[i].current_opinion,
            initial_opinion=agents[i].initial_opinion,
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
    agents: list[Agent],
    topic: str,
    steps: int = DEFAULT_STEPS,
    seed: Optional[int] = None,
    log_path: Optional[str | Path] = None,
    show_progress: bool = True,
    return_state: bool = False,
    return_side_labels: bool = False,
    persona_set: str = "personas",
) -> (
    tuple[list[float], list[float], list[float], list[dict[str, int]], dict[str, int], dict[str, int]]
    | tuple[list[float], list[float], list[float], list[dict[str, int]], list[list[str]], dict[str, int], dict[str, int]]
    | tuple[list[float], list[float], list[float], list[dict[str, int]], dict[str, int], dict[str, int], nx.Graph, list[Agent]]
    | tuple[list[float], list[float], list[float], list[dict[str, int]], list[list[str]], dict[str, int], dict[str, int], nx.Graph, list[Agent]]
):
    # Identify the bot by its node attribute (added by main.py calling add_bot)
    bot_id = next((i for i, d in G.nodes(data=True) if d.get("is_bot")), None)
    if bot_id is None:
        raise ValueError("Bot not found in graph. add_bot() must be called before run_with_bot_on_graph.")

    from tqdm import tqdm

    variances: list[float] = []
    polarizations: list[float] = []
    drifts: list[float] = []
    side_counts: list[dict[str, int]] = []
    side_labels_over_time: list[list[str]] = []

    model = _get_model(show_progress=False)
    real_agents = [a for a in agents if not a.is_bot]
    init_personas = [a.persona_prompt for a in real_agents]
    init_embs = model.encode(init_personas, convert_to_numpy=True, show_progress_bar=False)
    norms_i = np.linalg.norm(init_embs, axis=1, keepdims=True)
    norms_i = np.where(norms_i == 0, 1e-9, norms_i)
    init_embs_norm = init_embs / norms_i

    blocks = [G.nodes[a.node_id].get("block", 0) for a in real_agents]

    def _collect_votes(agent_list: list[Agent], desc_str: str) -> dict[str, int]:
        votes = {"SUPPORT": 0, "AGAINST": 0, "ABSTAIN": 0}
        agent_iter = tqdm(agent_list, desc=desc_str, unit="agent") if show_progress else agent_list
        for a in agent_iter:
            ans = get_vote(a.persona_prompt, topic, a.current_opinion)
            if ans in votes:
                votes[ans] += 1
            else:
                votes["ABSTAIN"] += 1
        return votes

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
            "bot_deploy_steps": list(BOT_DEPLOY_STEPS),
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
    initial_drift = mean_persona_drift(curr_embs, init_embs_norm * norms_i) # reconstruct raw init_embs
    drifts.append(initial_drift)
    
    # Initial classification
    labels0 = classify_side_labels(emb0, persona_set=persona_set)
    side_counts.append({k: labels0.count(k) for k in PERSONA_BLOCKS})
    if return_side_labels:
        side_labels_over_time.append(labels0)

    initial_votes = _collect_votes(real_agents, "Voting (Initial)")

    step_range = range(1, steps + 1)
    if show_progress:
        step_range = tqdm(step_range, desc="Intervention", unit="step")
    for t in step_range:
        step_t0 = perf_counter()
        step_stats = step_semantic_with_bot(G, agents, topic, bot_id, t)
        opinions = [a.current_opinion for a in agents]
        emb = embed_opinions(opinions)
        variances.append(semantic_variance(emb))
        polarizations.append(opinion_polarization(emb[:len(real_agents)], blocks))
        
        curr_personas = [a.persona_prompt for a in real_agents]
        curr_embs = model.encode(curr_personas, convert_to_numpy=True, show_progress_bar=False)
        drifts.append(mean_persona_drift(curr_embs, init_embs))
        
        # Classification
        labels = classify_side_labels(emb, persona_set=persona_set)
        side_counts.append({k: labels.count(k) for k in PERSONA_BLOCKS})
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
                "opinion_polarization": float(polarizations[-1]),
                "persona_drift_mean": float(drifts[-1]),
                "side_counts": side_counts[-1],
                "llm_updates": llm_updates,
                "avg_neighbors_used": float(step_stats["avg_neighbors_used"]),
                "avg_neighbor_chars": float(step_stats["avg_neighbor_chars"]),
                "bot_amplified_updates": bot_amplified_updates,
                "bot_deployed": bool(step_stats["bot_deployed"]),
                "step_elapsed_sec": round(perf_counter() - step_t0, 4),
            }
            log_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    final_votes = _collect_votes(real_agents, "Voting (Final)")

    if log_fh is not None:
        final = {
            "type": "run_summary",
            "total_llm_updates": int(total_llm_updates),
            "total_bot_amplified_updates": int(total_bot_amplified_updates),
            "total_elapsed_sec": round(perf_counter() - run_t0, 4),
            "initial_votes": initial_votes,
            "final_votes": final_votes,
        }
        log_fh.write(json.dumps(final, ensure_ascii=False) + "\n")
        log_fh.close()
    if return_state:
        if return_side_labels:
            return variances, polarizations, drifts, side_counts, side_labels_over_time, initial_votes, final_votes, G, agents
        return variances, polarizations, drifts, side_counts, initial_votes, final_votes, G, agents
    if return_side_labels:
        return variances, polarizations, drifts, side_counts, side_labels_over_time, initial_votes, final_votes
    return variances, polarizations, drifts, side_counts, initial_votes, final_votes
