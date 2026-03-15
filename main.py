#!/usr/bin/env python3
"""
Single-entry CLI for final experiments.

Run one condition:
    python main.py run --graph {er|rgglr} --bot {off|on}

Run full comparison matrix:
    python main.py matrix
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

MPL_CACHE_DIR = Path(tempfile.gettempdir()) / "ece227_matplotlib_cache"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import Normalize

from src.config import (
    DEFAULT_BOT_POST_PROB,
    DEFAULT_ER_EDGE_PROB,
    DEFAULT_LOG_MODE,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_TOPIC,
    LONG_RANGE_FRACTION,
    LONG_RANGE_K,
    MAX_CHARS_PER_NEIGHBOR,
    MAX_NEIGHBORS_PER_UPDATE,
    PERSONA_BLOCKS,
    RGG_RADIUS,
    SEED_LIST,
)

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _save_plot(run_dir: Path, filename: str) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / filename


def _make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _make_experiment_dir(mode: str, *parts: str) -> tuple[str, Path]:
    stamp = _make_run_id()
    clean_parts = [p.replace(" ", "_").replace("/", "-") for p in parts if p]
    folder = "_".join([mode, *clean_parts, stamp])
    path = OUTPUT_DIR / folder
    path.mkdir(parents=True, exist_ok=True)
    return stamp, path


def _node_positions(G: nx.Graph, seed: int = DEFAULT_SEED) -> dict[int, tuple[float, float]]:
    has_pos = all("pos" in G.nodes[n] for n in G.nodes())
    if has_pos:
        return {int(n): tuple(G.nodes[n]["pos"]) for n in G.nodes()}
    return nx.spring_layout(G, seed=seed)


def _plot_topology(G: nx.Graph, out_path: Path, title: str, seed: int = DEFAULT_SEED) -> None:
    pos = _node_positions(G, seed=seed)

    side_colors = {
        "democrat": "#3b82f6",
        "republican": "#ef4444",
        "independent": "#22c55e",
        "bot": "#111827",
        "unknown": "#9ca3af",
    }
    node_colors = [side_colors.get(str(G.nodes[n].get("side", "unknown")), "#9ca3af") for n in G.nodes()]

    has_edge_type = any("edge_type" in d for _, _, d in G.edges(data=True))
    local_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") != "long_range"]
    long_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "long_range"]

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_edges(G, pos, edgelist=local_edges, alpha=0.28, edge_color="#9ca3af", width=1.0)
    if has_edge_type and long_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=long_edges,
            alpha=0.6,
            edge_color="#8b5cf6",
            style="dashed",
            width=1.3,
        )
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=230, linewidths=0.5, edgecolors="white")

    deg = dict(G.degree())
    label_nodes = [n for n, _ in sorted(deg.items(), key=lambda x: x[1], reverse=True)[:8]]
    labels = {n: str(G.nodes[n].get("name", n)).split("_")[-1] for n in label_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=7)

    plt.title(title)
    plt.xlabel("Ideological Position")
    plt.ylabel("Engagement Level")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _opinion_drift_by_node(agents) -> dict[int, float]:
    from src.measurement import embed_opinions

    if not agents:
        return {}
    initial = [a.initial_opinion for a in agents]
    final = [a.current_opinion for a in agents]
    emb_i = embed_opinions(initial)
    emb_f = embed_opinions(final)

    norm_i = np.linalg.norm(emb_i, axis=1, keepdims=True)
    norm_f = np.linalg.norm(emb_f, axis=1, keepdims=True)
    cos_sim = np.sum(
        (emb_i / np.where(norm_i == 0, 1e-9, norm_i)) * (emb_f / np.where(norm_f == 0, 1e-9, norm_f)),
        axis=1,
    )
    drift = 1.0 - np.clip(cos_sim, -1.0, 1.0)
    return {int(a.node_id): float(d) for a, d in zip(agents, drift)}


def _plot_drift_network(
    G: nx.Graph,
    agents,
    out_path: Path,
    title: str,
    seed: int = DEFAULT_SEED,
) -> None:
    drift = _opinion_drift_by_node(agents)
    if not drift:
        return

    pos = _node_positions(G, seed=seed)
    nodes = list(G.nodes())
    vals = np.array([float(drift.get(int(n), 0.0)) for n in nodes], dtype=float)

    cmap = plt.cm.plasma
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    denom = (vmax - vmin) if (vmax > vmin) else 1.0
    colors = [cmap((v - vmin) / denom) for v in vals]

    local_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") != "long_range"]
    long_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "long_range"]

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_edges(G, pos, edgelist=local_edges, alpha=0.22, edge_color="#9ca3af", width=1.0, ax=ax)
    if long_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=long_edges,
            alpha=0.58,
            edge_color="#8b5cf6",
            style="dashed",
            width=1.2,
            ax=ax,
        )
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=colors,
        node_size=250,
        linewidths=0.5,
        edgecolors="white",
        ax=ax,
    )

    top_nodes = [n for n, _ in sorted(drift.items(), key=lambda x: x[1], reverse=True)[:10]]
    labels = {int(n): str(G.nodes[int(n)].get("name", n)).split("_")[-1] for n in top_nodes if int(n) in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Opinion Drift (cosine Δ from initial)")

    ax.set_title(title)
    ax.set_xlabel("Ideological Position")
    ax.set_ylabel("Engagement Level")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close()


def _write_run_timeseries_csv(
    out_path: Path,
    semantic_var: Optional[list[float]],
    side_counts: Optional[list[dict[str, int]]],
) -> None:
    if semantic_var is None or len(semantic_var) == 0:
        return

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "t",
                "semantic_variance",
                "democrat_count",
                "republican_count",
                "independent_count",
            ],
        )
        writer.writeheader()
        for t in range(len(semantic_var)):
            counts = side_counts[t] if side_counts is not None and t < len(side_counts) else {}
            writer.writerow(
                {
                    "t": t,
                    "semantic_variance": semantic_var[t],
                    "democrat_count": counts.get("democrat"),
                    "republican_count": counts.get("republican"),
                    "independent_count": counts.get("independent"),
                }
            )


def _plot_vote_comparison(
    initial_votes: dict[str, int],
    final_votes: dict[str, int],
    out_path: Path,
    title: str,
) -> None:
    labels = ["SUPPORT", "AGAINST", "ABSTAIN"]
    init_counts = [initial_votes.get(l, 0) for l in labels]
    final_counts = [final_votes.get(l, 0) for l in labels]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width/2, init_counts, width, label='Initial Vote', color='#60a5fa')
    rects2 = ax.bar(x + width/2, final_counts, width, label='Final Vote', color='#f472b6')
    
    ax.set_ylabel('Number of Nodes')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()

def _build_graph(graph_key: str, seed: int, persona_set: str = "personas"):
    from src.graphs.er import create_er_graph
    from src.graphs.rgg_long_range import RGGLongRangeParams, create_rgg_long_range_graph
    from src.load_nodes import load_nodes

    if graph_key == "er":
        return create_er_graph(edge_prob=DEFAULT_ER_EDGE_PROB, seed=seed, persona_set=persona_set), "ER"

    nodes = load_nodes(persona_set)
    params = RGGLongRangeParams(
        radius=RGG_RADIUS,
        long_range_fraction=LONG_RANGE_FRACTION,
        long_range_k=LONG_RANGE_K,
        seed=seed,
    )
    return create_rgg_long_range_graph(nodes, params), "RGGLR"


def _graph_structure_metrics(G) -> dict[str, float | int]:
    n_nodes = int(G.number_of_nodes())
    n_edges = int(G.number_of_edges())
    blocks = [G.nodes[i].get("block", 0) for i in range(n_nodes)]

    degs = np.array([d for _, d in G.degree()], dtype=float)
    avg_degree = float(np.mean(degs)) if degs.size else 0.0
    min_degree = int(np.min(degs)) if degs.size else 0
    max_degree = int(np.max(degs)) if degs.size else 0
    isolates = int(np.sum(degs == 0)) if degs.size else 0

    density = float(nx.density(G)) if n_nodes > 1 else 0.0
    components = int(nx.number_connected_components(G)) if n_nodes > 0 else 0

    has_edge_type = any("edge_type" in d for _, _, d in G.edges(data=True))
    if has_edge_type:
        long_range_edges = int(sum(1 for _, _, d in G.edges(data=True) if d.get("edge_type") == "long_range"))
        local_edges = n_edges - long_range_edges
    else:
        long_range_edges = 0
        local_edges = n_edges

    # Structural
    avg_clustering = nx.average_clustering(G)
    transitivity = nx.transitivity(G)

    try:
        avg_sp = nx.average_shortest_path_length(G)
        diameter = nx.diameter(G)
    except nx.NetworkXError:
        # Not connected - use largest component
        if n_nodes > 0:
            lcc = max(nx.connected_components(G), key=len)
            H = G.subgraph(lcc)
            avg_sp = nx.average_shortest_path_length(H)
            diameter = nx.diameter(H)
        else:
            avg_sp = 0.0
            diameter = 0

    try:
        deg_assort = nx.degree_assortativity_coefficient(G)
    except Exception:
        deg_assort = float("nan")

    # Modularity using block partition
    block_communities = {}
    for i, b in enumerate(blocks):
        block_communities.setdefault(b, set()).add(i)
    communities = list(block_communities.values())
    try:
        modularity = nx.community.modularity(G, communities)
    except Exception:
        modularity = float("nan")

    # Bridge edges (cut edges)
    bridge_edges = set(nx.bridges(G))
    bridge_fraction = len(bridge_edges) / max(G.number_of_edges(), 1)

    # Centrality
    betweenness = nx.betweenness_centrality(G, normalized=True)
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {i: 0.0 for i in G.nodes()}

    avg_betweenness = float(np.mean(list(betweenness.values()))) if betweenness else 0.0
    avg_eigenvector = float(np.mean(list(eigenvector.values()))) if eigenvector else 0.0

    return {
        "nodes": n_nodes,
        "edges": n_edges,
        "density": density,
        "components": components,
        "isolates": isolates,
        "avg_degree": avg_degree,
        "min_degree": min_degree,
        "max_degree": max_degree,
        "local_edges": local_edges,
        "long_range_edges": long_range_edges,
        "avg_clustering": float(avg_clustering),
        "transitivity": float(transitivity),
        "avg_shortest_path": float(avg_sp),
        "diameter": int(diameter),
        "degree_assortativity": float(deg_assort),
        "modularity": float(modularity),
        "bridge_edge_fraction": float(bridge_fraction),
        "avg_betweenness": avg_betweenness,
        "avg_eigenvector": avg_eigenvector,
    }


def main_run(args: argparse.Namespace) -> dict[str, list[float]]:
    from src.intervention import run_with_bot_on_graph
    from src.simulation import create_agents, run_semantic
    from src.degroot import run_degroot
    from src.config import PERSONA_BLOCK_LAYOUT, side_from_name

    assert args.graph in ("er", "rgglr"), "run requires --graph {er|rgglr}"
    assert args.bot in ("off", "on"), "run requires --bot {off|on}"
    persona_set = getattr(args, "persona_set", None)
    assert persona_set in ("personas", "senate"), "run requires --persona-set {personas|senate}"
    seed = getattr(args, "seed", None)
    if seed is None:
        seed = DEFAULT_SEED
    args.seed = seed

    run_stamp, run_dir = _make_experiment_dir(
        "run",
        args.graph,
        "semantic",
        f"bot-{args.bot}",
        f"seed-{args.seed}",
    )
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    G, graph_label = _build_graph(args.graph, args.seed, persona_set=persona_set)

    run_id = f"{graph_label}_{'bot' if args.bot == "on" else "no_bot"}"
    print(f"Running {run_id} | topic={DEFAULT_TOPIC} | persona_set={persona_set}")
    print(f"Graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, seed={args.seed}")
    print(f"Output folder: {run_dir}")

    log_path = None
    if not args.no_log:
        log_path = logs_dir / "step_summary.jsonl"
        print(f"Logging compact {DEFAULT_LOG_MODE} records to {log_path}")

    semantic_var: list[float] | None = None
    semantic_pol: list[float] | None = None
    semantic_drift: list[float] | None = None
    side_counts: list[dict[str, int]] | None = None
    semantic_graph = G
    semantic_agents = None

    # Run DeGroot baseline
    print(f"\n[{run_id}] DeGroot (baseline)...")
    n = G.number_of_nodes()
    initial_degroot = np.zeros(n)
    for i in range(n):
        node_name = G.nodes[i].get("name", "")
        side = side_from_name(node_name)
        if side in PERSONA_BLOCK_LAYOUT:
            initial_degroot[i] = PERSONA_BLOCK_LAYOUT[side][1]
        else:
            initial_degroot[i] = 0.5
            
    degroot_history = run_degroot(G, initial_degroot, steps=DEFAULT_STEPS)
    degroot_var = [float(np.var(x)) for x in degroot_history]
    print("DeGroot variance over time:")
    for t, v in enumerate(degroot_var):
        print(f"  t={t}: {v:.4f}")

    if True:  # always run semantic
        if args.bot == "on":
            semantic_var, semantic_pol, semantic_drift, side_counts, initial_votes, final_votes, semantic_graph, semantic_agents = run_with_bot_on_graph(
                G=G,
                topic=DEFAULT_TOPIC,
                steps=DEFAULT_STEPS,
                bot_post_prob=DEFAULT_BOT_POST_PROB,
                seed=args.seed,
                log_path=log_path,
                show_progress=True,
                return_state=True,
                persona_set=persona_set,
            )
        else:
            agents = create_agents(G, topic=DEFAULT_TOPIC)
            semantic_var, semantic_pol, semantic_drift, side_counts, initial_votes, final_votes = run_semantic(
                G=G,
                agents=agents,
                topic=DEFAULT_TOPIC,
                steps=DEFAULT_STEPS,
                log_path=log_path,
                persona_set=persona_set,
            )
            semantic_agents = agents
            semantic_graph = G
        print("\nSemantic variance over time:")
        for t, v in enumerate(semantic_var):
            print(f"  t={t}: {v:.4f}")

    display_graph = semantic_graph if args.bot == "on" and semantic_graph is not None else G
    topology_path = _save_plot(run_dir, "network_topology.png")
    _plot_topology(
        display_graph,
        topology_path,
        title=f"{run_id}: Network Topology",
        seed=args.seed,
    )
    print(f"Saved {topology_path}")

    if degroot_var is not None:
        out_deg = _save_plot(run_dir, "degroot_variance.png")
        plt.figure()
        plt.plot(degroot_var, marker="s", color="orange")
        plt.xlabel("Timestep")
        plt.ylabel("DeGroot Variance")
        plt.title(f"{run_id}: DeGroot Variance")
        plt.grid(True)
        plt.savefig(out_deg, dpi=160)
        plt.close()
        print(f"Saved {out_deg}")

    if semantic_var is not None:
        out = _save_plot(run_dir, "semantic_variance.png")
        plt.figure()
        plt.plot(semantic_var, marker="o")
        plt.xlabel("Timestep")
        plt.ylabel("Semantic Variance")
        plt.title(f"{run_id}: Semantic Variance")
        plt.grid(True)
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"Saved {out}")
        
    if semantic_pol is not None:
        out_pol = _save_plot(run_dir, "semantic_polarization.png")
        plt.figure()
        plt.plot(semantic_pol, marker="o", color="purple")
        plt.xlabel("Timestep")
        plt.ylabel("Opinion Polarization")
        plt.title(f"{run_id}: Opinion Polarization Over Time")
        plt.grid(True)
        plt.savefig(out_pol, dpi=160)
        plt.close()
        print(f"Saved {out_pol}")

    if semantic_drift is not None:
        out_drift = _save_plot(run_dir, "persona_drift.png")
        plt.figure()
        plt.plot(semantic_drift, marker="o", color="red")
        plt.xlabel("Timestep")
        plt.ylabel("Persona Drift Mean")
        plt.title(f"{run_id}: Persona Drift Over Time")
        plt.grid(True)
        plt.savefig(out_drift, dpi=160)
        plt.close()
        print(f"Saved {out_drift}")

    if side_counts is not None:
        out2 = _save_plot(run_dir, "side_counts.png")
        steps = list(range(len(side_counts)))
        labels = [s for s in PERSONA_BLOCKS if s in side_counts[0]]
        plt.figure()
        for lab in labels:
            series = [c[lab] for c in side_counts]
            plt.plot(steps, series, marker="o", label=lab)
        plt.xlabel("Timestep")
        plt.ylabel("Count of agents")
        plt.title(f"{run_id}: Side Counts")
        plt.legend()
        plt.grid(True)
        plt.savefig(out2, dpi=160)
        plt.close()
        print(f"Saved {out2}")

    if semantic_agents is not None:
        drift_path = _save_plot(run_dir, "opinion_drift_network.png")
        _plot_drift_network(
            semantic_graph,
            semantic_agents,
            drift_path,
            title=f"{run_id}: Opinion Drift by Node",
            seed=args.seed,
        )
        print(f"Saved {drift_path}")

    timeseries_path = run_dir / "timeseries.csv"
    _write_run_timeseries_csv(timeseries_path, semantic_var, side_counts)
    print(f"Saved {timeseries_path}")

    if semantic_agents is not None and initial_votes is not None and final_votes is not None:
        vote_plot_path = run_dir / "vote_comparison.png"
        _plot_vote_comparison(
            initial_votes,
            final_votes,
            vote_plot_path,
            title=f"{run_id}: Initial vs Final Votes",
        )
        print(f"Saved {vote_plot_path}")

    summary = {
        "run_id": run_id,
        "timestamp": run_stamp,
        "output_dir": str(run_dir),
        "graph": args.graph,
        "graph_label": graph_label,
        "model": "semantic",
        "bot": args.bot,
        "seed": args.seed,
        "topic": DEFAULT_TOPIC,
        "steps": DEFAULT_STEPS,
        "config": {
            "edge_prob": DEFAULT_ER_EDGE_PROB,
            "bot_post_prob": DEFAULT_BOT_POST_PROB,
            "rgg_radius": RGG_RADIUS,
            "rgg_long_range_fraction": LONG_RANGE_FRACTION,
            "rgg_long_range_k": LONG_RANGE_K,
            "log_mode": DEFAULT_LOG_MODE,
            "max_neighbors_per_update": MAX_NEIGHBORS_PER_UPDATE,
            "max_chars_per_neighbor": MAX_CHARS_PER_NEIGHBOR,
        },
        "graph_metrics": _graph_structure_metrics(display_graph),
    }
    if semantic_var is not None:
        summary["semantic_final_variance"] = float(semantic_var[-1])
        summary["initial_votes"] = initial_votes
        summary["final_votes"] = final_votes
    if degroot_var is not None:
        summary["degroot_final_variance"] = float(degroot_var[-1])

    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {summary_path}")

    out: dict[str, list[float]] = {}
    if semantic_var is not None:
        out["semantic"] = semantic_var
    return out


def _matrix_log_path(
    matrix_dir: Path,
    graph_key: str,
    persona_set: str,
    model: str,
    bot: str,
    seed: int,
    enabled: bool,
) -> Optional[Path]:
    if not enabled:
        return None
    log_dir = matrix_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{graph_key}_{persona_set}_{model}_{bot}_seed{seed}_summary.jsonl"


def _append_matrix_rows(
    rows: list[dict[str, object]],
    *,
    matrix_id: str,
    graph: str,
    persona_set: str,
    model: str,
    bot: str,
    seed: int,
    steps: int,
    topic: str,
    variances: list[float],
    polarizations: list[float] | None,
    drifts: list[float] | None,
    side_counts: Optional[list[dict[str, int]]],
    initial_votes: dict[str, int] | None,
    final_votes: dict[str, int] | None,
    graph_metrics: dict[str, float | int],
    bot_degree: Optional[int],
) -> None:
    if len(variances) != steps + 1:
        raise ValueError(
            f"Expected {steps + 1} variance points but got {len(variances)} for {graph}/{persona_set}/{model}/{bot}/seed={seed}."
        )

    v0 = float(variances[0])
    prev: Optional[float] = None

    for t, variance in enumerate(variances):
        counts = side_counts[t] if side_counts is not None else None
        democrat = counts.get("democrat", 0) if counts else None
        republican = counts.get("republican", 0) if counts else None
        independent = counts.get("independent", 0) if counts else None
        measured_agents = (democrat + republican + independent) if counts else None

        votes_dict = None
        if t == 0 and initial_votes is not None:
            votes_dict = initial_votes
        elif t == steps and final_votes is not None:
            votes_dict = final_votes

        row = {
            "matrix_id": matrix_id,
            "graph": graph,
            "persona_set": persona_set,
            "model": model,
            "bot": bot,
            "seed": seed,
            "t": t,
            "topic": topic,
            "variance": float(variance),
            "polarization": float(polarizations[t]) if polarizations else None,
            "persona_drift_mean": float(drifts[t]) if drifts else None,
            "delta_from_t0": float(variance) - v0,
            "delta_from_prev": None if prev is None else float(variance) - prev,
            "democrat_count": democrat,
            "republican_count": republican,
            "independent_count": independent,
            "measured_agents": measured_agents,
            "vote_support_count": votes_dict.get("SUPPORT", 0) if votes_dict else None,
            "vote_against_count": votes_dict.get("AGAINST", 0) if votes_dict else None,
            "vote_abstain_count": votes_dict.get("ABSTAIN", 0) if votes_dict else None,
            "graph_nodes": graph_metrics["nodes"],
            "graph_edges": graph_metrics["edges"],
            "graph_density": graph_metrics["density"],
            "graph_components": graph_metrics["components"],
            "graph_isolates": graph_metrics["isolates"],
            "graph_avg_degree": graph_metrics["avg_degree"],
            "graph_min_degree": graph_metrics["min_degree"],
            "graph_max_degree": graph_metrics["max_degree"],
            "graph_local_edges": graph_metrics["local_edges"],
            "graph_long_range_edges": graph_metrics["long_range_edges"],
            "graph_avg_clustering": graph_metrics.get("avg_clustering"),
            "graph_transitivity": graph_metrics.get("transitivity"),
            "graph_avg_shortest_path": graph_metrics.get("avg_shortest_path"),
            "graph_diameter": graph_metrics.get("diameter"),
            "graph_degree_assortativity": graph_metrics.get("degree_assortativity"),
            "graph_modularity": graph_metrics.get("modularity"),
            "graph_bridge_edge_fraction": graph_metrics.get("bridge_edge_fraction"),
            "graph_avg_betweenness": graph_metrics.get("avg_betweenness"),
            "graph_avg_eigenvector": graph_metrics.get("avg_eigenvector"),
            "bot_degree": bot_degree,
        }
        rows.append(row)
        prev = float(variance)


def _write_matrix_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    if not rows:
        raise ValueError("No matrix rows generated.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "matrix_id",
        "graph",
        "persona_set",
        "model",
        "bot",
        "seed",
        "t",
        "topic",
        "variance",
        "polarization",
        "persona_drift_mean",
        "delta_from_t0",
        "delta_from_prev",
        "democrat_count",
        "republican_count",
        "independent_count",
        "measured_agents",
        "vote_support_count",
        "vote_against_count",
        "vote_abstain_count",
        "graph_nodes",
        "graph_edges",
        "graph_density",
        "graph_components",
        "graph_isolates",
        "graph_avg_degree",
        "graph_min_degree",
        "graph_max_degree",
        "graph_local_edges",
        "graph_long_range_edges",
        "graph_avg_clustering",
        "graph_transitivity",
        "graph_avg_shortest_path",
        "graph_diameter",
        "graph_degree_assortativity",
        "graph_modularity",
        "graph_bridge_edge_fraction",
        "graph_avg_betweenness",
        "graph_avg_eigenvector",
        "bot_degree",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_matrix_summary(rows: list[dict[str, object]], final_t: int) -> dict[str, dict[str, float | int]]:
    buckets: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if int(row["t"]) != final_t:
            continue
        key = (str(row["graph"]), str(row["persona_set"]), str(row["bot"]))
        buckets[key].append(float(row["variance"]))

    summary: dict[str, dict[str, float | int]] = {}
    print("\nFinal-step variance summary (mean +/- std across seeds):")
    for key in sorted(buckets.keys()):
        values = np.array(buckets[key], dtype=float)
        k = f"{key[0]}|{key[1]}|{key[2]}"
        summary[k] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "n": int(len(values)),
        }
        print(
            f"  {key[0]} | {key[1]} | bot={key[2]} -> "
            f"{float(np.mean(values)):.4f} +/- {float(np.std(values)):.4f} (n={len(values)})"
        )
    return summary


def _plot_matrix_condition_lines(rows: list[dict[str, object]], out_path: Path) -> None:
    by_condition: dict[tuple[str, str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cond = (str(row["graph"]), str(row["persona_set"]), str(row["bot"]))
        t = int(row["t"])
        by_condition[cond][t].append(float(row["variance"]))

    plt.figure(figsize=(11, 6))
    for cond in sorted(by_condition.keys()):
        ts = sorted(by_condition[cond].keys())
        means = [float(np.mean(by_condition[cond][t])) for t in ts]
        label = f"{cond[0]} | {cond[1]} | bot={cond[2]}"
        plt.plot(ts, means, marker="o", linewidth=2, label=label)
    plt.xlabel("Timestep")
    plt.ylabel("Mean Variance Across Seeds")
    plt.title("Matrix: Condition Variance Trajectories")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _condition_label(cond: tuple[str, str, str]) -> str:
    graph, persona_set, bot = cond
    return f"{graph} | {persona_set} | bot={bot}"


def _plot_matrix_final_step_bars(rows: list[dict[str, object]], out_path: Path, final_t: int) -> None:
    by_condition: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if int(row["t"]) != final_t:
            continue
        cond = (str(row["graph"]), str(row["persona_set"]), str(row["bot"]))
        by_condition[cond].append(float(row["variance"]))

    conds = sorted(by_condition.keys())
    means = [float(np.mean(by_condition[c])) for c in conds]
    stds = [float(np.std(by_condition[c])) for c in conds]
    labels = [_condition_label(c) for c in conds]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(conds))
    plt.bar(x, means, yerr=stds, capsize=5, color="#60a5fa", edgecolor="#1e3a8a", alpha=0.85)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel(f"Variance at t={final_t}")
    plt.title("Matrix: Final-Step Variance (mean +/- std across seeds)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_matrix_variance_heatmap(rows: list[dict[str, object]], out_path: Path) -> None:
    by_condition_t: dict[tuple[str, str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cond = (str(row["graph"]), str(row["persona_set"]), str(row["bot"]))
        t = int(row["t"])
        by_condition_t[cond][t].append(float(row["variance"]))

    conds = sorted(by_condition_t.keys())
    ts = sorted({int(row["t"]) for row in rows})
    mat = np.zeros((len(conds), len(ts)), dtype=float)

    for i, cond in enumerate(conds):
        for j, t in enumerate(ts):
            vals = by_condition_t[cond].get(t, [])
            mat[i, j] = float(np.mean(vals)) if vals else np.nan

    plt.figure(figsize=(12, 6))
    im = plt.imshow(mat, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="Mean Variance Across Seeds")
    plt.yticks(np.arange(len(conds)), [_condition_label(c) for c in conds])
    plt.xticks(np.arange(len(ts)), ts)
    plt.xlabel("Timestep")
    plt.ylabel("Condition")
    plt.title("Matrix: Variance Heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_matrix_bot_effect(rows: list[dict[str, object]], out_path: Path) -> None:
    series: dict[tuple[str, str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        graph = str(row["graph"])
        persona_set = str(row["persona_set"])
        bot = str(row["bot"])
        if str(row["model"]) != "semantic":
            continue
        t = int(row["t"])
        key = (graph, persona_set, bot)
        series[key][t].append(float(row["variance"]))

    graph_persona = sorted({(g, p) for g, p, _ in series.keys()})
    plt.figure(figsize=(11, 6))
    for graph, persona_set in graph_persona:
        on = series.get((graph, persona_set, "on"), {})
        off = series.get((graph, persona_set, "off"), {})
        ts = sorted(set(on.keys()) & set(off.keys()))
        if not ts:
            continue
        effects: list[float] = []
        for t in ts:
            mean_on = float(np.mean(on[t]))
            mean_off = float(np.mean(off[t]))
            effects.append(mean_on - mean_off)
        plt.plot(ts, effects, marker="o", linewidth=2, label=f"{graph} | {persona_set}: bot on - off")

    plt.axhline(0.0, color="#374151", linewidth=1.0, linestyle="--")
    plt.xlabel("Timestep")
    plt.ylabel("Variance Difference")
    plt.title("Matrix: Bot Effect Over Time")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _side_entropy(democrat: int, republican: int, independent: int) -> float:
    counts = np.array([democrat, republican, independent], dtype=float)
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    probs = counts / total
    nonzero = probs[probs > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def _plot_matrix_side_counts(rows: list[dict[str, object]], out_path: Path) -> None:
    """Mean democrat/republican/independent counts over time by condition."""
    by_cond_t: dict[tuple[str, str, str], dict[int, list[tuple[int, int, int]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if str(row["model"]) != "semantic":
            continue
        if row["democrat_count"] is None:
            continue
        cond = (str(row["graph"]), str(row["persona_set"]), str(row["bot"]))
        t = int(row["t"])
        by_cond_t[cond][t].append((
            int(row["democrat_count"]),
            int(row["republican_count"]),
            int(row["independent_count"]),
        ))
    conds = sorted(by_cond_t.keys())
    ncols = 4
    nrows = (len(conds) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows), squeeze=False)
    axes = axes.flatten()
    colors = {"democrat": "#3b82f6", "republican": "#ef4444", "independent": "#22c55e"}
    for idx, cond in enumerate(conds):
        ax = axes[idx]
        ts = sorted(by_cond_t[cond].keys())
        dem = [float(np.mean([x[0] for x in by_cond_t[cond][t]])) for t in ts]
        rep = [float(np.mean([x[1] for x in by_cond_t[cond][t]])) for t in ts]
        ind = [float(np.mean([x[2] for x in by_cond_t[cond][t]])) for t in ts]
        ax.plot(ts, dem, marker="o", markersize=4, color=colors["democrat"], label="democrat")
        ax.plot(ts, rep, marker="o", markersize=4, color=colors["republican"], label="republican")
        ax.plot(ts, ind, marker="o", markersize=4, color=colors["independent"], label="independent")
        ax.set_title(f"{cond[0]} | {cond[1]} | bot={cond[2]}")
        ax.set_xlabel("t")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Matrix: Side Counts Over Time by Condition", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_matrix_side_entropy(rows: list[dict[str, object]], out_path: Path) -> None:
    by_condition_t: dict[tuple[str, str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if str(row["model"]) != "semantic":
            continue
        if row["democrat_count"] is None:
            continue
        graph = str(row["graph"])
        persona_set = str(row["persona_set"])
        bot = str(row["bot"])
        t = int(row["t"])
        entropy = _side_entropy(
            int(row["democrat_count"]),
            int(row["republican_count"]),
            int(row["independent_count"]),
        )
        by_condition_t[(graph, persona_set, bot)][t].append(entropy)

    plt.figure(figsize=(11, 6))
    for cond in sorted(by_condition_t.keys()):
        ts = sorted(by_condition_t[cond].keys())
        means = [float(np.mean(by_condition_t[cond][t])) for t in ts]
        label = f"{cond[0]} | {cond[1]} | bot={cond[2]}"
        plt.plot(ts, means, marker="o", linewidth=2, label=label)

    plt.xlabel("Timestep")
    plt.ylabel("Side Entropy (bits)")
    plt.title("Matrix: Semantic Side-Mix Entropy")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _accumulate_side_transitions(
    side_labels_over_time: list[list[str]],
    transition_counts: np.ndarray,
) -> None:
    side_index = {side: idx for idx, side in enumerate(PERSONA_BLOCKS)}
    for t in range(len(side_labels_over_time) - 1):
        src_labels = side_labels_over_time[t]
        dst_labels = side_labels_over_time[t + 1]
        n = min(len(src_labels), len(dst_labels))
        for i in range(n):
            src = src_labels[i]
            dst = dst_labels[i]
            if src in side_index and dst in side_index:
                transition_counts[side_index[src], side_index[dst]] += 1.0


def _accumulate_final_side_transitions(
    side_labels_over_time: list[list[str]],
    transition_counts: np.ndarray,
) -> None:
    if len(side_labels_over_time) < 2:
        return
    side_index = {side: idx for idx, side in enumerate(PERSONA_BLOCKS)}
    src_labels = side_labels_over_time[0]
    dst_labels = side_labels_over_time[-1]
    n = min(len(src_labels), len(dst_labels))
    for i in range(n):
        src = src_labels[i]
        dst = dst_labels[i]
        if src in side_index and dst in side_index:
            transition_counts[side_index[src], side_index[dst]] += 1.0


def _accumulate_transition_timing(
    side_labels_over_time: list[list[str]],
    changed_counts: np.ndarray,
    total_counts: np.ndarray,
) -> None:
    side_index = {side: idx for idx, side in enumerate(PERSONA_BLOCKS)}
    for t in range(len(side_labels_over_time) - 1):
        src_labels = side_labels_over_time[t]
        dst_labels = side_labels_over_time[t + 1]
        n = min(len(src_labels), len(dst_labels))
        for i in range(n):
            src = src_labels[i]
            dst = dst_labels[i]
            if src in side_index and dst in side_index:
                total_counts[t] += 1.0
                if src != dst:
                    changed_counts[t] += 1.0


def _plot_side_transition_timing_bot_on_off(
    changed_counts_by_bot: dict[str, np.ndarray],
    total_counts_by_bot: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    plt.figure(figsize=(11, 6))
    for bot in ("off", "on"):
        changed = changed_counts_by_bot.get(bot)
        total = total_counts_by_bot.get(bot)
        if changed is None or total is None or len(changed) == 0:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            rates = np.divide(changed, total, where=total > 0) * 100.0
        ts = np.arange(1, len(rates) + 1)
        plt.plot(ts, rates, marker="o", linewidth=2, label=f"bot={bot}")

    plt.xlabel("Transition Step (t -> t+1)")
    plt.ylabel("Agents Changing Side (%)")
    plt.title("Matrix: Side Transition Timing by Bot Condition")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_side_final_transition_matrix_bot_on_off(
    final_transitions_by_bot: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    bots = ("off", "on")
    mats = [
        final_transitions_by_bot.get(bot, np.zeros((len(PERSONA_BLOCKS), len(PERSONA_BLOCKS))))
        for bot in bots
    ]
    norm_mats: list[np.ndarray] = []
    for mat in mats:
        row_sum = mat.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = np.divide(mat, row_sum, where=row_sum > 0)
        norm_mats.append(norm)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    vmax = max(float(np.max(norm)) for norm in norm_mats) if norm_mats else 1.0
    vmax = max(vmax, 1e-6)
    im = None
    for ax, bot, raw_mat, norm_mat in zip(axes, bots, mats, norm_mats):
        im = ax.imshow(norm_mat, cmap="YlOrRd", vmin=0.0, vmax=vmax)
        ax.set_title(f"bot={bot}")
        ax.set_xlabel("Final side (t=final)")
        ax.set_ylabel("Initial side (t=0)")
        ax.set_xticks(np.arange(len(PERSONA_BLOCKS)))
        ax.set_yticks(np.arange(len(PERSONA_BLOCKS)))
        ax.set_xticklabels(PERSONA_BLOCKS, rotation=25, ha="right")
        ax.set_yticklabels(PERSONA_BLOCKS)
        for i in range(len(PERSONA_BLOCKS)):
            for j in range(len(PERSONA_BLOCKS)):
                pct = norm_mat[i, j] * 100.0
                cnt = int(raw_mat[i, j])
                text_color = "white" if norm_mat[i, j] > 0.55 * vmax else "#111827"
                ax.text(j, i, f"{pct:.0f}%\n(n={cnt})", ha="center", va="center", fontsize=8, color=text_color)

    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.85, label="Probability from Initial Side")
    fig.suptitle("Matrix: Initial -> Final Side Transition by Bot Condition", fontsize=13)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _write_transition_summary_csv(
    out_path: Path,
    transitions_by_persona_by_bot: dict[str, dict[str, np.ndarray]],
    final_transitions_by_persona_by_bot: dict[str, dict[str, np.ndarray]],
    changed_counts_by_persona_by_bot: dict[str, dict[str, np.ndarray]],
    total_counts_by_persona_by_bot: dict[str, dict[str, np.ndarray]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["persona_set", "bot", "metric", "value"]
    rows: list[dict[str, object]] = []
    for persona_set in sorted(transitions_by_persona_by_bot.keys()):
        transitions_by_bot = transitions_by_persona_by_bot[persona_set]
        final_transitions_by_bot = final_transitions_by_persona_by_bot[persona_set]
        changed_counts_by_bot = changed_counts_by_persona_by_bot[persona_set]
        total_counts_by_bot = total_counts_by_persona_by_bot[persona_set]
        for bot in ("off", "on"):
            step_mat = transitions_by_bot.get(bot, np.zeros((len(PERSONA_BLOCKS), len(PERSONA_BLOCKS))))
            final_mat = final_transitions_by_bot.get(bot, np.zeros((len(PERSONA_BLOCKS), len(PERSONA_BLOCKS))))
            total_step = float(np.sum(step_mat))
            total_final = float(np.sum(final_mat))
            step_stay = float(np.trace(step_mat) / total_step) if total_step > 0 else 0.0
            final_stay = float(np.trace(final_mat) / total_final) if total_final > 0 else 0.0
            changed = changed_counts_by_bot.get(bot, np.zeros(0, dtype=float))
            totals = total_counts_by_bot.get(bot, np.zeros(0, dtype=float))
            with np.errstate(divide="ignore", invalid="ignore"):
                rates = np.divide(changed, totals, where=totals > 0)
            mean_rate = float(np.mean(rates)) if len(rates) else 0.0
            peak_rate = float(np.max(rates)) if len(rates) else 0.0
            peak_step = int(np.argmax(rates) + 1) if len(rates) else 0
            rows.extend(
                [
                    {"persona_set": persona_set, "bot": bot, "metric": "step_stay_rate", "value": round(step_stay, 6)},
                    {"persona_set": persona_set, "bot": bot, "metric": "initial_to_final_stay_rate", "value": round(final_stay, 6)},
                    {"persona_set": persona_set, "bot": bot, "metric": "mean_step_change_rate", "value": round(mean_rate, 6)},
                    {"persona_set": persona_set, "bot": bot, "metric": "peak_step_change_rate", "value": round(peak_rate, 6)},
                    {"persona_set": persona_set, "bot": bot, "metric": "peak_step_change_rate_step", "value": peak_step},
                ]
            )

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_side_transition_matrix_bot_on_off(
    transitions_by_bot: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    bots = ("off", "on")
    mats = [transitions_by_bot.get(bot, np.zeros((len(PERSONA_BLOCKS), len(PERSONA_BLOCKS)))) for bot in bots]
    norm_mats: list[np.ndarray] = []
    for mat in mats:
        row_sum = mat.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = np.divide(mat, row_sum, where=row_sum > 0)
        norm_mats.append(norm)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    vmax = max(float(np.max(norm)) for norm in norm_mats) if norm_mats else 1.0
    vmax = max(vmax, 1e-6)
    im = None
    for ax, bot, raw_mat, norm_mat in zip(axes, bots, mats, norm_mats):
        im = ax.imshow(norm_mat, cmap="Blues", vmin=0.0, vmax=vmax)
        ax.set_title(f"bot={bot}")
        ax.set_xlabel("To side (t+1)")
        ax.set_ylabel("From side (t)")
        ax.set_xticks(np.arange(len(PERSONA_BLOCKS)))
        ax.set_yticks(np.arange(len(PERSONA_BLOCKS)))
        ax.set_xticklabels(PERSONA_BLOCKS, rotation=25, ha="right")
        ax.set_yticklabels(PERSONA_BLOCKS)
        for i in range(len(PERSONA_BLOCKS)):
            for j in range(len(PERSONA_BLOCKS)):
                pct = norm_mat[i, j] * 100.0
                cnt = int(raw_mat[i, j])
                text_color = "white" if norm_mat[i, j] > 0.55 * vmax else "#111827"
                ax.text(j, i, f"{pct:.0f}%\n(n={cnt})", ha="center", va="center", fontsize=8, color=text_color)

    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.85, label="Transition Probability")
    fig.suptitle("Matrix: Side Transition (t -> t+1) by Bot Condition", fontsize=13)
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_matrix_analysis_pack(
    rows: list[dict[str, object]],
    out_dir: Path,
    final_t: int,
    transitions_by_persona_by_bot: dict[str, dict[str, np.ndarray]],
    final_transitions_by_persona_by_bot: dict[str, dict[str, np.ndarray]],
    changed_counts_by_persona_by_bot: dict[str, dict[str, np.ndarray]],
    total_counts_by_persona_by_bot: dict[str, dict[str, np.ndarray]],
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [
        out_dir / "final_step_variance_bars.png",
        out_dir / "variance_heatmap.png",
        out_dir / "bot_effect_over_time.png",
        out_dir / "side_counts_over_time.png",
        out_dir / "semantic_side_entropy.png",
    ]
    _plot_matrix_final_step_bars(rows, files[0], final_t=final_t)
    _plot_matrix_variance_heatmap(rows, files[1])
    _plot_matrix_bot_effect(rows, files[2])
    _plot_matrix_side_counts(rows, files[3])
    _plot_matrix_side_entropy(rows, files[4])
    for persona_set in sorted(transitions_by_persona_by_bot.keys()):
        t_by_bot = transitions_by_persona_by_bot[persona_set]
        f_by_bot = final_transitions_by_persona_by_bot[persona_set]
        c_by_bot = changed_counts_by_persona_by_bot[persona_set]
        ttot_by_bot = total_counts_by_persona_by_bot[persona_set]
        files.append(out_dir / f"side_transition_matrix_bot_on_off_{persona_set}.png")
        _plot_side_transition_matrix_bot_on_off(t_by_bot, files[-1])
        files.append(out_dir / f"side_transition_timing_bot_on_off_{persona_set}.png")
        _plot_side_transition_timing_bot_on_off(c_by_bot, ttot_by_bot, files[-1])
        files.append(out_dir / f"side_transition_matrix_initial_to_final_bot_on_off_{persona_set}.png")
        _plot_side_final_transition_matrix_bot_on_off(f_by_bot, files[-1])
    return files


def main_matrix(args: argparse.Namespace) -> dict[str, object]:
    from src.intervention import add_bot, run_with_bot_on_graph
    from src.simulation import create_agents, run_semantic
    from src.degroot import run_degroot
    from src.config import PERSONA_BLOCK_LAYOUT, side_from_name

    matrix_id, matrix_dir = _make_experiment_dir(
        "matrix",
        "er-rgglr",
        "personas-senate",
        f"seeds-{len(SEED_LIST)}",
        f"steps-{DEFAULT_STEPS}",
    )
    rows: list[dict[str, object]] = []
    matrix_graphs = ("er", "rgglr")
    matrix_persona_sets = ("personas", "senate")
    matrix_steps = DEFAULT_STEPS
    matrix_topic = DEFAULT_TOPIC
    matrix_bot_prob = DEFAULT_BOT_POST_PROB

    def _init_transitions() -> dict[str, np.ndarray]:
        return {
            "off": np.zeros((len(PERSONA_BLOCKS), len(PERSONA_BLOCKS)), dtype=float),
            "on": np.zeros((len(PERSONA_BLOCKS), len(PERSONA_BLOCKS)), dtype=float),
        }

    def _init_timing(steps: int) -> dict[str, np.ndarray]:
        return {
            "off": np.zeros(steps, dtype=float),
            "on": np.zeros(steps, dtype=float),
        }

    transitions_by_persona_by_bot: dict[str, dict[str, np.ndarray]] = {
        p: _init_transitions() for p in matrix_persona_sets
    }
    final_transitions_by_persona_by_bot: dict[str, dict[str, np.ndarray]] = {
        p: _init_transitions() for p in matrix_persona_sets
    }
    changed_counts_by_persona_by_bot: dict[str, dict[str, np.ndarray]] = {
        p: _init_timing(matrix_steps) for p in matrix_persona_sets
    }
    total_counts_by_persona_by_bot: dict[str, dict[str, np.ndarray]] = {
        p: _init_timing(matrix_steps) for p in matrix_persona_sets
    }

    print(
        "Running matrix (canonical config): "
        f"graphs={','.join(matrix_graphs)} | persona_sets={','.join(matrix_persona_sets)} | "
        f"seeds={SEED_LIST} | steps={matrix_steps} | topic={matrix_topic}"
    )
    print(f"Output folder: {matrix_dir}")

    rep_seed = SEED_LIST[0]
    for graph_key in matrix_graphs:
        for persona_set in matrix_persona_sets:
            G_top, graph_label = _build_graph(graph_key, rep_seed, persona_set=persona_set)
            topo_path = matrix_dir / f"network_topology_{graph_key}_{persona_set}.png"
            _plot_topology(
                G_top,
                topo_path,
                title=f"Matrix: {graph_label} | {persona_set} (seed={rep_seed})",
                seed=rep_seed,
            )
            print(f"Saved {topo_path}")

    for graph_key in matrix_graphs:
        for persona_set in matrix_persona_sets:
            for seed in SEED_LIST:
                G, graph_label = _build_graph(graph_key, seed, persona_set=persona_set)
                base_metrics = _graph_structure_metrics(G)

                print(f"\n[{graph_label} {persona_set} seed={seed}] DeGroot (baseline)...")
                n = G.number_of_nodes()
                initial_degroot = np.zeros(n)
                for i in range(n):
                    node_name = G.nodes[i].get("name", "")
                    side = side_from_name(node_name)
                    if side in PERSONA_BLOCK_LAYOUT:
                        initial_degroot[i] = PERSONA_BLOCK_LAYOUT[side][1]
                    else:
                        initial_degroot[i] = 0.5
                
                degroot_history = run_degroot(G, initial_degroot, steps=matrix_steps)
                degroot_var = [float(np.var(x)) for x in degroot_history]
                
                _append_matrix_rows(
                    rows,
                    matrix_id=matrix_id,
                    graph=graph_key,
                    persona_set=persona_set,
                    model="degroot",
                    bot="off",
                    seed=seed,
                    steps=matrix_steps,
                    topic=matrix_topic,
                    variances=degroot_var,
                    polarizations=None,
                    drifts=None,
                    side_counts=None,
                    initial_votes=None,
                    final_votes=None,
                    graph_metrics=base_metrics,
                    bot_degree=None,
                )

                print(f"[{graph_label} {persona_set} seed={seed}] Semantic (no bot)...")
                semantic_log_path = _matrix_log_path(
                    matrix_dir,
                    graph_key,
                    persona_set,
                    model="semantic",
                    bot="off",
                    seed=seed,
                    enabled=args.log_runs,
                )
                agents = create_agents(G, topic=matrix_topic)
                semantic_var, semantic_pol, semantic_drift, semantic_counts, semantic_labels, initial_votes, final_votes = run_semantic(
                    G=G,
                    agents=agents,
                    topic=matrix_topic,
                    steps=matrix_steps,
                    show_progress=args.show_progress,
                    log_path=semantic_log_path,
                    return_side_labels=True,
                    persona_set=persona_set,
                )
                _accumulate_side_transitions(semantic_labels, transitions_by_persona_by_bot[persona_set]["off"])
                _accumulate_final_side_transitions(semantic_labels, final_transitions_by_persona_by_bot[persona_set]["off"])
                _accumulate_transition_timing(
                    semantic_labels,
                    changed_counts_by_persona_by_bot[persona_set]["off"],
                    total_counts_by_persona_by_bot[persona_set]["off"],
                )
                _append_matrix_rows(
                    rows,
                    matrix_id=matrix_id,
                    graph=graph_key,
                    persona_set=persona_set,
                    model="semantic",
                    bot="off",
                    seed=seed,
                    steps=matrix_steps,
                    topic=matrix_topic,
                    variances=semantic_var,
                    polarizations=semantic_pol,
                    drifts=semantic_drift,
                    side_counts=semantic_counts,
                    initial_votes=initial_votes,
                    final_votes=final_votes,
                    graph_metrics=base_metrics,
                    bot_degree=None,
                )
                
                drift_path_off = matrix_dir / f"opinion_drift_network_{graph_key}_{persona_set}_seed-{seed}_bot-off.png"
                _plot_drift_network(
                    G,
                    agents,
                    drift_path_off,
                    title=f"Drift: {graph_label} | {persona_set} (seed={seed}) | Bot: off",
                    seed=seed,
                )

                print(f"[{graph_label} {persona_set} seed={seed}] Semantic (+ bot)...")
                bot_metric_agents = create_agents(G, topic=matrix_topic)
                G_with_bot, _ = add_bot(G, bot_metric_agents, seed=seed)
                bot_metrics = _graph_structure_metrics(G_with_bot)
                bot_degree = int(G_with_bot.degree(G_with_bot.number_of_nodes() - 1))

                bot_log_path = _matrix_log_path(
                    matrix_dir,
                    graph_key,
                    persona_set,
                    model="semantic",
                    bot="on",
                    seed=seed,
                    enabled=args.log_runs,
                )
                semantic_bot_var, semantic_bot_pol, semantic_bot_drift, semantic_bot_counts, semantic_bot_labels, initial_votes_bot, final_votes_bot, G_bot_ret, bot_agents_ret = run_with_bot_on_graph(
                    G=G,
                    topic=matrix_topic,
                    steps=matrix_steps,
                    bot_post_prob=matrix_bot_prob,
                    seed=seed,
                    log_path=bot_log_path,
                    show_progress=args.show_progress,
                    return_side_labels=True,
                    return_state=True,
                    persona_set=persona_set,
                )
                _accumulate_side_transitions(semantic_bot_labels, transitions_by_persona_by_bot[persona_set]["on"])
                _accumulate_final_side_transitions(semantic_bot_labels, final_transitions_by_persona_by_bot[persona_set]["on"])
                _accumulate_transition_timing(
                    semantic_bot_labels,
                    changed_counts_by_persona_by_bot[persona_set]["on"],
                    total_counts_by_persona_by_bot[persona_set]["on"],
                )
                _append_matrix_rows(
                    rows,
                    matrix_id=matrix_id,
                    graph=graph_key,
                    persona_set=persona_set,
                    model="semantic",
                    bot="on",
                    seed=seed,
                    steps=matrix_steps,
                    topic=matrix_topic,
                    variances=semantic_bot_var,
                    polarizations=semantic_bot_pol,
                    drifts=semantic_bot_drift,
                    side_counts=semantic_bot_counts,
                    initial_votes=initial_votes_bot,
                    final_votes=final_votes_bot,
                    graph_metrics=bot_metrics,
                    bot_degree=bot_degree,
                )

                drift_path_on = matrix_dir / f"opinion_drift_network_{graph_key}_{persona_set}_seed-{seed}_bot-on.png"
                _plot_drift_network(
                    G_bot_ret,
                    bot_agents_ret,
                    drift_path_on,
                    title=f"Drift: {graph_label} | {persona_set} (seed={seed}) | Bot: on",
                    seed=seed,
                )

    out_path = matrix_dir / "matrix_results.csv"
    _write_matrix_csv(rows, out_path)

    print(f"\nMatrix complete. Wrote {len(rows)} rows to {out_path}")
    summary = _print_matrix_summary(rows, final_t=matrix_steps)

    condition_plot = matrix_dir / "condition_variance_trajectories.png"
    _plot_matrix_condition_lines(rows, condition_plot)
    print(f"Saved {condition_plot}")

    analysis_plots = _plot_matrix_analysis_pack(
        rows,
        matrix_dir,
        final_t=matrix_steps,
        transitions_by_persona_by_bot=transitions_by_persona_by_bot,
        final_transitions_by_persona_by_bot=final_transitions_by_persona_by_bot,
        changed_counts_by_persona_by_bot=changed_counts_by_persona_by_bot,
        total_counts_by_persona_by_bot=total_counts_by_persona_by_bot,
    )
    for p in analysis_plots:
        print(f"Saved {p}")

    transition_summary_path = matrix_dir / "side_transition_summary.csv"
    _write_transition_summary_csv(
        transition_summary_path,
        transitions_by_persona_by_bot=transitions_by_persona_by_bot,
        final_transitions_by_persona_by_bot=final_transitions_by_persona_by_bot,
        changed_counts_by_persona_by_bot=changed_counts_by_persona_by_bot,
        total_counts_by_persona_by_bot=total_counts_by_persona_by_bot,
    )
    print(f"Saved {transition_summary_path}")

    summary_path = matrix_dir / "matrix_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "matrix_id": matrix_id,
                "output_dir": str(matrix_dir),
                "graphs": list(matrix_graphs),
                "persona_sets": list(matrix_persona_sets),
                "seeds": SEED_LIST,
                "steps": matrix_steps,
                "topic": matrix_topic,
                "final_step_summary": summary,
                "transition_summary_csv": str(transition_summary_path),
                "rows": len(rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {summary_path}")

    if args.out:
        custom_path = Path(args.out)
        custom_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_path, custom_path)
        print(f"Copied matrix CSV to {custom_path}")

    return {
        "matrix_id": matrix_id,
        "out_path": str(Path(args.out) if args.out else out_path),
        "rows": len(rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one condition (`run`) or full cross-condition sweep (`matrix`)."
    )
    sub = parser.add_subparsers(dest="mode")

    p_run = sub.add_parser("run", help="Run one canonical condition (semantic/SBERT only)")
    p_run.add_argument("--graph", choices=["er", "rgglr"], required=True, help="Graph structure: er or rgglr")
    p_run.add_argument("--bot", choices=["off", "on"], required=True, help="Bot: off or on")
    p_run.add_argument(
        "--persona-set",
        choices=["personas", "senate"],
        required=True,
        help="Node file: personas (data/nodes.json) or senate (data/senate_nodes.json)",
    )
    p_run.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})",
    )
    p_run.add_argument(
        "--no-log",
        action="store_true",
        help="Disable writing compact per-step log in <run_folder>/logs/step_summary.jsonl",
    )
    p_run.set_defaults(func=main_run)

    p_matrix = sub.add_parser("matrix", help="Run canonical ER/RGGLR x DeGroot/Semantic x bot/off matrix")
    p_matrix.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional extra copy path for matrix CSV (primary CSV is always saved in the run folder).",
    )
    p_matrix.add_argument("--show-progress", action="store_true", help="Show tqdm progress bars for semantic runs")
    p_matrix.add_argument(
        "--log-runs",
        action="store_true",
        help="Write compact per-run summary logs under <matrix_run_folder>/logs/",
    )
    p_matrix.set_defaults(func=main_matrix)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.mode is None:
        parser.print_help()
        return 1
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
