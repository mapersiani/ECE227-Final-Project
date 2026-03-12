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
from src.measurement import embed_opinions, semantic_variance
from src.network import (
    create_erdos_renyi_graph,
    create_scale_free_graph,
    create_small_world_rgg_graph,
)
from src.simulation import create_agents, create_agents_from_nodes_data, run_semantic
from src.config import DEFAULT_ER_EDGE_PROB

def _load_bot_profile(topic: str) -> tuple[str, str]:
    """
    Load bot persona and initial opinion from src/trump_env_reg.json.
    No built-in fallback: file and required fields must exist.
    """
    _ = topic
    path = Path(__file__).resolve().parent / "trump_env_reg.json"
    if not path.exists():
        raise FileNotFoundError(f"Bot profile file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    prompt = str(payload.get("prompt", "")).strip()
    initial = str(payload.get("initial", "")).strip()
    if not prompt or not initial:
        raise ValueError(f"Bot profile must contain non-empty 'prompt' and 'initial': {path}")
    return prompt, initial


def add_bot(
    G: nx.Graph,
    agents: list[Agent],
    topic: str,
    bot_post_prob: float = 0.8,
    connect_prob: float = 0.2,
    seed: Optional[int] = 42,
    target_nodes: Optional[list[int]] = None,
) -> tuple[nx.Graph, list[Agent]]:
    """
    Add bot node connected to ~20% of existing nodes. Bot opinion is fixed.
    """
    n = G.number_of_nodes()
    bot_id = n
    G = G.copy()
    rng = np.random.default_rng(seed)
    if target_nodes is not None:
        for i in target_nodes:
            if 0 <= i < n:
                G.add_edge(i, bot_id)
    else:
        for i in range(n):
            if rng.random() < connect_prob:
                G.add_edge(i, bot_id)
    bot_persona, bot_text = _load_bot_profile(topic)
    bot = Agent(node_id=bot_id, persona_prompt=bot_persona, initial_opinion=bot_text, is_bot=True)
    bot.update_opinion(bot_text)
    return G, list(agents) + [bot]


def step_semantic_with_bot(
    G: nx.Graph,
    agents: list[Agent],
    topic: str,
    bot_id: int,
    bot_post_prob: float = 0.8,
    bot_post_multiplier: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """
    One step. Bot neighbors see bot opinion with probability bot_post_prob (simulates high frequency).
    """
    opinions = [a.current_opinion for a in agents]
    if rng is None:
        rng = np.random.default_rng()
    n = G.number_of_nodes()
    for i in range(n):
        if i == bot_id:
            continue
        neighbors = list(G.neighbors(i))
        neighbor_opinions = [opinions[j] for j in neighbors if j != bot_id]
        if bot_id in neighbors:
            # bot_post_prob means "posts" vs "silent" at this step.
            repeats = bot_post_multiplier if rng.random() < bot_post_prob else 0
            if repeats > 0:
                neighbor_opinions.extend([opinions[bot_id]] * repeats)
        if not neighbor_opinions:
            continue
        new = get_updated_opinion(
            persona=agents[i].persona_prompt,
            topic=topic,
            neighbor_opinions=neighbor_opinions,
            memory=opinions[i],
        )
        agents[i].update_opinion(new)


def run_with_bot(
    topic: str = "AI Regulation",
    steps: int = 5,
    bot_post_prob: float = 0.8,
    bot_post_multiplier: int = 3,
    bot_connect_prob: float = 0.2,
    seed: Optional[int] = None,
) -> list[float]:
    """
    Run semantic simulation with bot. Returns semantic variance at t=0 through t=steps.
    """
    G = create_erdos_renyi_graph(n=20, p=DEFAULT_ER_EDGE_PROB, seed=seed)
    agents = create_agents(G, topic=topic, seed=seed)
    G, agents = add_bot(
        G,
        agents,
        topic=topic,
        bot_post_prob=bot_post_prob,
        connect_prob=bot_connect_prob,
        seed=seed,
    )
    bot_id = G.number_of_nodes() - 1

    variances = []
    opinions = [a.current_opinion for a in agents]
    variances.append(semantic_variance(embed_opinions(opinions)))
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        step_semantic_with_bot(
            G,
            agents,
            topic,
            bot_id,
            bot_post_prob=bot_post_prob,
            bot_post_multiplier=bot_post_multiplier,
            rng=rng,
        )
        opinions = [a.current_opinion for a in agents]
        variances.append(semantic_variance(embed_opinions(opinions)))
    return variances


def _run_with_bot_on_graph(
    G: nx.Graph,
    agents: list[Agent],
    topic: str,
    steps: int,
    bot_post_prob: float,
    bot_post_multiplier: int,
    bot_connect_prob: float,
    seed: Optional[int],
    bot_target_nodes: Optional[list[int]] = None,
) -> tuple[list[float], list[list[str]]]:
    """
    Run semantic dynamics with a fixed-opinion bot on a provided graph.
    Returns variance series (excluding bot node) and opinion history (excluding bot).
    """
    G_bot, agents_bot = add_bot(
        G,
        agents,
        topic=topic,
        bot_post_prob=bot_post_prob,
        connect_prob=bot_connect_prob,
        seed=seed,
        target_nodes=bot_target_nodes,
    )
    bot_id = G_bot.number_of_nodes() - 1

    history: list[list[str]] = [[a.current_opinion for a in agents_bot[:-1]]]
    var_series = [semantic_variance(embed_opinions(history[0]))]
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        step_semantic_with_bot(
            G_bot,
            agents_bot,
            topic,
            bot_id,
            bot_post_prob=bot_post_prob,
            bot_post_multiplier=bot_post_multiplier,
            rng=rng,
        )
        opinions_no_bot = [a.current_opinion for a in agents_bot[:-1]]
        history.append(opinions_no_bot)
        var_series.append(semantic_variance(embed_opinions(opinions_no_bot)))
    return var_series, history


def _select_bot_targets_by_degree(
    G: nx.Graph,
    mode: str,
    frac: float,
) -> list[int]:
    """
    Select target nodes for bot edges based on node degree.
    mode: "hub" (highest degree) or "periphery" (lowest degree)
    """
    n = G.number_of_nodes()
    k = max(1, int(round(n * frac)))
    degree_items = list(G.degree())
    if mode == "hub":
        sorted_nodes = sorted(degree_items, key=lambda x: (-x[1], x[0]))
    elif mode == "periphery":
        sorted_nodes = sorted(degree_items, key=lambda x: (x[1], x[0]))
    else:
        raise ValueError(f"unknown mode: {mode}")
    return [node for node, _deg in sorted_nodes[:k]]


def run_topology_resilience_with_bot(
    *,
    nodes_data: list[dict],
    topic: str,
    steps: int,
    seed: int,
    er_p: float,
    bot_post_prob: float,
    bot_post_multiplier: int,
    bot_connect_prob: float,
) -> dict:
    """
    Evaluate ER resilience to bot-induced semantic drift.

    Resilience proxy:
      - per-step semantic drift between baseline and with-bot trajectories
      - final drift and cumulative drift (smaller = more resilient)
    """
    n = len(nodes_data)
    G = create_erdos_renyi_graph(n=n, p=er_p, seed=seed)
    agents_base = create_agents_from_nodes_data(G, nodes_data, topic=topic)
    base_history = [[a.current_opinion for a in agents_base]]

    def _capture(_t, agents_state):
        base_history.append([a.current_opinion for a in agents_state])

    base_var = run_semantic(
        G,
        agents_base,
        topic,
        steps=steps,
        on_step=_capture,
        show_progress=True,
    )

    agents_for_bot = create_agents_from_nodes_data(G, nodes_data, topic=topic)
    bot_var, bot_history = _run_with_bot_on_graph(
        G,
        agents_for_bot,
        topic=topic,
        steps=steps,
        bot_post_prob=bot_post_prob,
        bot_post_multiplier=bot_post_multiplier,
        bot_connect_prob=bot_connect_prob,
        seed=seed,
    )

    drift_series: list[float] = []
    for t in range(len(base_history)):
        emb_base = embed_opinions(base_history[t])
        emb_bot = embed_opinions(bot_history[t])
        drift_series.append(float(np.mean(np.linalg.norm(emb_bot - emb_base, axis=1))))

    return {
        "er": {
            "n_nodes": n,
            "n_edges": G.number_of_edges(),
            "baseline_variance": base_var,
            "bot_variance": bot_var,
            "drift_series": drift_series,
            "final_drift": drift_series[-1],
            "cumulative_drift": float(np.sum(drift_series)),
        }
    }


def run_sw_sf_resilience_with_bot(
    *,
    nodes_data: list[dict],
    topic: str,
    steps: int,
    seed: int,
    sw_radius: float,
    sw_long_prob: float,
    sf_m: int,
    bot_post_prob: float,
    bot_post_multiplier: int,
    bot_connect_prob: float,
) -> dict:
    """
    Compare resilience under disinformation bot:
      - small_world: RGG + random long-distance connections
      - scale_free: Barabasi-Albert graph
    """
    n = len(nodes_data)
    topo_builders = {
        "small_world": lambda: create_small_world_rgg_graph(
            n=n, radius=sw_radius, long_edge_prob=sw_long_prob, seed=seed
        ),
        "scale_free": lambda: create_scale_free_graph(n=n, m=sf_m, seed=seed),
    }
    out: dict[str, dict] = {}
    for name, build in topo_builders.items():
        G = build()
        agents_base = create_agents_from_nodes_data(G, nodes_data, topic=topic)
        base_history = [[a.current_opinion for a in agents_base]]

        def _capture(_t, agents_state):
            base_history.append([a.current_opinion for a in agents_state])

        base_var = run_semantic(
            G,
            agents_base,
            topic,
            steps=steps,
            on_step=_capture,
            show_progress=True,
        )

        agents_for_bot = create_agents_from_nodes_data(G, nodes_data, topic=topic)
        bot_var, bot_history = _run_with_bot_on_graph(
            G,
            agents_for_bot,
            topic=topic,
            steps=steps,
            bot_post_prob=bot_post_prob,
            bot_post_multiplier=bot_post_multiplier,
            bot_connect_prob=bot_connect_prob,
            seed=seed,
        )

        drift_series: list[float] = []
        for t in range(len(base_history)):
            emb_base = embed_opinions(base_history[t])
            emb_bot = embed_opinions(bot_history[t])
            drift_series.append(float(np.mean(np.linalg.norm(emb_bot - emb_base, axis=1))))

        out[name] = {
            "n_nodes": n,
            "n_edges": G.number_of_edges(),
            "baseline_variance": base_var,
            "bot_variance": bot_var,
            "drift_series": drift_series,
            "final_drift": drift_series[-1],
            "cumulative_drift": float(np.sum(drift_series)),
        }
    return out


def run_hub_vs_periphery_bot_experiment(
    *,
    nodes_data: list[dict],
    topic: str,
    steps: int,
    seed: int,
    sf_m: int,
    bot_post_prob: float,
    bot_post_multiplier: int,
    bot_target_frac: float,
) -> dict:
    """
    Compare bot impact when connected to hubs vs periphery nodes on scale-free topology.
    """
    n = len(nodes_data)
    topology = "scale_free"
    G = create_scale_free_graph(n=n, m=sf_m, seed=seed)

    agents_base = create_agents_from_nodes_data(G, nodes_data, topic=topic)
    base_history = [[a.current_opinion for a in agents_base]]

    def _capture(_t, agents_state):
        base_history.append([a.current_opinion for a in agents_state])

    base_var = run_semantic(
        G,
        agents_base,
        topic,
        steps=steps,
        on_step=_capture,
        show_progress=True,
    )

    target_hub = _select_bot_targets_by_degree(G, mode="hub", frac=bot_target_frac)
    target_periphery = _select_bot_targets_by_degree(G, mode="periphery", frac=bot_target_frac)

    def _run_case(target_nodes: list[int]) -> dict:
        agents_case = create_agents_from_nodes_data(G, nodes_data, topic=topic)
        case_var, case_history = _run_with_bot_on_graph(
            G,
            agents_case,
            topic=topic,
            steps=steps,
            bot_post_prob=bot_post_prob,
            bot_post_multiplier=bot_post_multiplier,
            bot_connect_prob=0.0,
            seed=seed,
            bot_target_nodes=target_nodes,
        )
        drift_series: list[float] = []
        for t in range(len(base_history)):
            emb_base = embed_opinions(base_history[t])
            emb_case = embed_opinions(case_history[t])
            drift_series.append(float(np.mean(np.linalg.norm(emb_case - emb_base, axis=1))))
        return {
            "bot_variance": case_var,
            "drift_series": drift_series,
            "final_drift": drift_series[-1],
            "cumulative_drift": float(np.sum(drift_series)),
            "target_count": len(target_nodes),
        }

    hub_case = _run_case(target_hub)
    per_case = _run_case(target_periphery)

    return {
        "topology": topology,
        "n_nodes": n,
        "n_edges": G.number_of_edges(),
        "baseline_variance": base_var,
        "hub": hub_case,
        "periphery": per_case,
    }


def run_hub_vs_periphery_multiseed(
    *,
    nodes_data: list[dict],
    topic: str,
    steps: int,
    seeds: list[int],
    sf_m: int,
    bot_post_prob: float,
    bot_post_multiplier: int,
    bot_target_frac: float,
) -> dict:
    """
    Multi-seed aggregation for hub vs periphery bot placement experiment.
    Returns per-seed results and mean/std summaries.
    """
    runs: list[dict] = []
    for seed in seeds:
        run = run_hub_vs_periphery_bot_experiment(
            nodes_data=nodes_data,
            topic=topic,
            steps=steps,
            seed=seed,
            sf_m=sf_m,
            bot_post_prob=bot_post_prob,
            bot_post_multiplier=bot_post_multiplier,
            bot_target_frac=bot_target_frac,
        )
        run["seed"] = seed
        runs.append(run)

    hub_final = np.array([r["hub"]["final_drift"] for r in runs], dtype=float)
    per_final = np.array([r["periphery"]["final_drift"] for r in runs], dtype=float)
    hub_cum = np.array([r["hub"]["cumulative_drift"] for r in runs], dtype=float)
    per_cum = np.array([r["periphery"]["cumulative_drift"] for r in runs], dtype=float)

    return {
        "topology": "scale_free",
        "seeds": list(seeds),
        "runs": runs,
        "summary": {
            "hub_final_mean": float(np.mean(hub_final)),
            "hub_final_std": float(np.std(hub_final)),
            "per_final_mean": float(np.mean(per_final)),
            "per_final_std": float(np.std(per_final)),
            "hub_cum_mean": float(np.mean(hub_cum)),
            "hub_cum_std": float(np.std(hub_cum)),
            "per_cum_mean": float(np.mean(per_cum)),
            "per_cum_std": float(np.std(per_cum)),
        },
    }
