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
    DEGROOT_SCALAR_BY_BLOCK,
    DEFAULT_BOT_POST_PROB,
    DEFAULT_ER_EDGE_PROB,
    DEFAULT_LOG_MODE,
    DEFAULT_N,
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
        "left": "#3b82f6",
        "center_left": "#22c55e",
        "center_right": "#f59e0b",
        "right": "#ef4444",
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

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_edges(G, pos, edgelist=local_edges, alpha=0.22, edge_color="#9ca3af", width=1.0)
    if long_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=long_edges,
            alpha=0.58,
            edge_color="#8b5cf6",
            style="dashed",
            width=1.2,
        )
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=250, linewidths=0.5, edgecolors="white")

    top_nodes = [n for n, _ in sorted(drift.items(), key=lambda x: x[1], reverse=True)[:10]]
    labels = {int(n): str(G.nodes[int(n)].get("name", n)).split("_")[-1] for n in top_nodes if int(n) in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=7)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, label="Opinion Drift (cosine Δ from initial)")

    plt.title(title)
    plt.xlabel("Ideological Position")
    plt.ylabel("Engagement Level")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _write_run_timeseries_csv(
    out_path: Path,
    semantic_var: Optional[list[float]],
    degroot_var: Optional[list[float]],
    side_counts: Optional[list[dict[str, int]]],
) -> None:
    max_len = 0
    if semantic_var is not None:
        max_len = max(max_len, len(semantic_var))
    if degroot_var is not None:
        max_len = max(max_len, len(degroot_var))
    if max_len == 0:
        return

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "t",
                "semantic_variance",
                "degroot_variance",
                "left_count",
                "center_left_count",
                "center_right_count",
                "right_count",
            ],
        )
        writer.writeheader()
        for t in range(max_len):
            counts = side_counts[t] if side_counts is not None and t < len(side_counts) else {}
            writer.writerow(
                {
                    "t": t,
                    "semantic_variance": semantic_var[t] if semantic_var is not None and t < len(semantic_var) else None,
                    "degroot_variance": degroot_var[t] if degroot_var is not None and t < len(degroot_var) else None,
                    "left_count": counts.get("left"),
                    "center_left_count": counts.get("center_left"),
                    "center_right_count": counts.get("center_right"),
                    "right_count": counts.get("right"),
                }
            )

def _build_graph(graph_key: str, seed: int):
    from src.graphs.er import create_er_graph
    from src.graphs.rgg_long_range import RGGLongRangeParams, create_rgg_long_range_graph
    from src.network import load_nodes

    if graph_key == "er":
        return create_er_graph(edge_prob=DEFAULT_ER_EDGE_PROB, seed=seed), "ER"

    nodes = load_nodes()
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
    }


def _degroot_variance_series(G, steps: int) -> list[float]:
    from src.network import run_degroot

    side_map = DEGROOT_SCALAR_BY_BLOCK
    n = G.number_of_nodes()
    sides = [G.nodes[i].get("side", "center_left") for i in range(n)]
    initial_scalar = np.array([side_map.get(s, 0.5) for s in sides], dtype=float)
    history = run_degroot(G, initial_scalar, steps=steps)
    return [float(np.var(h)) for h in history]


def main_run(args: argparse.Namespace) -> dict[str, list[float]]:
    from src.intervention import run_with_bot_on_graph
    from src.simulation import create_agents, run_semantic

    run_stamp, run_dir = _make_experiment_dir(
        "run",
        args.graph,
        args.model,
        f"bot-{args.bot}",
        f"seed-{args.seed}",
    )
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    G, graph_label = _build_graph(args.graph, args.seed)

    if G.number_of_nodes() != DEFAULT_N:
        raise ValueError(f"{graph_label} graph must initialize with {DEFAULT_N} nodes.")

    if args.bot == "on" and args.model in {"degroot", "both"}:
        raise ValueError("DeGroot comparison currently supports --bot off only.")

    run_id = f"{graph_label}_{'bot' if args.bot == 'on' else 'no_bot'}"
    print(f"Running {run_id} | model={args.model} | topic={DEFAULT_TOPIC}")
    print(f"Graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, seed={args.seed}")
    print(f"Output folder: {run_dir}")

    log_path = None
    if args.model in {"semantic", "both"} and not args.no_log:
        log_path = logs_dir / "step_summary.jsonl"
        print(f"Logging compact {DEFAULT_LOG_MODE} records to {log_path}")

    semantic_var: list[float] | None = None
    degroot_var: list[float] | None = None
    side_counts: list[dict[str, int]] | None = None
    semantic_graph = G
    semantic_agents = None

    if args.model in {"semantic", "both"}:
        if args.bot == "on":
            semantic_var, side_counts, semantic_graph, semantic_agents = run_with_bot_on_graph(
                G=G,
                topic=DEFAULT_TOPIC,
                steps=DEFAULT_STEPS,
                bot_post_prob=DEFAULT_BOT_POST_PROB,
                seed=args.seed,
                log_path=log_path,
                show_progress=True,
                return_state=True,
            )
        else:
            agents = create_agents(G, topic=DEFAULT_TOPIC)
            semantic_var, side_counts = run_semantic(
                G=G,
                agents=agents,
                topic=DEFAULT_TOPIC,
                steps=DEFAULT_STEPS,
                log_path=log_path,
            )
            semantic_agents = agents
            semantic_graph = G
        print("\nSemantic variance over time:")
        for t, v in enumerate(semantic_var):
            print(f"  t={t}: {v:.4f}")

    if args.model in {"degroot", "both"}:
        degroot_var = _degroot_variance_series(G, steps=DEFAULT_STEPS)
        print("\nDeGroot variance over time:")
        for t, v in enumerate(degroot_var):
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

    if degroot_var is not None:
        out = _save_plot(run_dir, "degroot_variance.png")
        plt.figure()
        plt.plot(degroot_var, marker="s", color="orange")
        plt.xlabel("Timestep")
        plt.ylabel("Opinion Variance")
        plt.title(f"{run_id}: DeGroot Variance")
        plt.grid(True)
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"Saved {out}")

    if semantic_var is not None and degroot_var is not None:
        out = _save_plot(run_dir, "semantic_vs_degroot.png")
        plt.figure()
        plt.plot(semantic_var, marker="o", label="Semantic (LLM)")
        plt.plot(degroot_var, marker="s", label="DeGroot")
        plt.xlabel("Timestep")
        plt.ylabel("Variance")
        plt.title(f"{run_id}: Semantic vs DeGroot")
        plt.legend()
        plt.grid(True)
        plt.savefig(out, dpi=160)
        plt.close()
        print(f"Saved {out}")

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
    _write_run_timeseries_csv(timeseries_path, semantic_var, degroot_var, side_counts)
    print(f"Saved {timeseries_path}")

    summary = {
        "run_id": run_id,
        "timestamp": run_stamp,
        "output_dir": str(run_dir),
        "graph": args.graph,
        "graph_label": graph_label,
        "model": args.model,
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
    if degroot_var is not None:
        summary["degroot_final_variance"] = float(degroot_var[-1])

    summary_path = run_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {summary_path}")

    out: dict[str, list[float]] = {}
    if semantic_var is not None:
        out["semantic"] = semantic_var
    if degroot_var is not None:
        out["degroot"] = degroot_var
    return out


def _matrix_log_path(
    matrix_dir: Path,
    graph_key: str,
    model: str,
    bot: str,
    seed: int,
    enabled: bool,
) -> Optional[Path]:
    if not enabled:
        return None
    log_dir = matrix_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{graph_key}_{model}_{bot}_seed{seed}_summary.jsonl"


def _append_matrix_rows(
    rows: list[dict[str, object]],
    *,
    matrix_id: str,
    graph: str,
    model: str,
    bot: str,
    seed: int,
    steps: int,
    topic: str,
    variances: list[float],
    side_counts: Optional[list[dict[str, int]]],
    graph_metrics: dict[str, float | int],
    bot_degree: Optional[int],
) -> None:
    if len(variances) != steps + 1:
        raise ValueError(
            f"Expected {steps + 1} variance points but got {len(variances)} for {graph}/{model}/{bot}/seed={seed}."
        )

    v0 = float(variances[0])
    prev: Optional[float] = None

    for t, variance in enumerate(variances):
        counts = side_counts[t] if side_counts is not None else None
        left = counts.get("left", 0) if counts else None
        center_left = counts.get("center_left", 0) if counts else None
        center_right = counts.get("center_right", 0) if counts else None
        right = counts.get("right", 0) if counts else None
        measured_agents = (left + center_left + center_right + right) if counts else None

        row = {
            "matrix_id": matrix_id,
            "graph": graph,
            "model": model,
            "bot": bot,
            "seed": seed,
            "t": t,
            "topic": topic,
            "variance": float(variance),
            "delta_from_t0": float(variance) - v0,
            "delta_from_prev": None if prev is None else float(variance) - prev,
            "left_count": left,
            "center_left_count": center_left,
            "center_right_count": center_right,
            "right_count": right,
            "measured_agents": measured_agents,
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
        "model",
        "bot",
        "seed",
        "t",
        "topic",
        "variance",
        "delta_from_t0",
        "delta_from_prev",
        "left_count",
        "center_left_count",
        "center_right_count",
        "right_count",
        "measured_agents",
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
        key = (str(row["graph"]), str(row["model"]), str(row["bot"]))
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
        cond = (str(row["graph"]), str(row["model"]), str(row["bot"]))
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
    graph, model, bot = cond
    return f"{graph} | {model} | bot={bot}"


def _plot_matrix_final_step_bars(rows: list[dict[str, object]], out_path: Path, final_t: int) -> None:
    by_condition: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if int(row["t"]) != final_t:
            continue
        cond = (str(row["graph"]), str(row["model"]), str(row["bot"]))
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
        cond = (str(row["graph"]), str(row["model"]), str(row["bot"]))
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
    series: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        graph = str(row["graph"])
        model = str(row["model"])
        bot = str(row["bot"])
        if model != "semantic":
            continue
        t = int(row["t"])
        key = (graph, bot)
        series[key][t].append(float(row["variance"]))

    graphs = sorted({g for g, _ in series.keys()})
    plt.figure(figsize=(11, 6))
    for graph in graphs:
        on = series.get((graph, "on"), {})
        off = series.get((graph, "off"), {})
        ts = sorted(set(on.keys()) & set(off.keys()))
        if not ts:
            continue
        effects: list[float] = []
        for t in ts:
            mean_on = float(np.mean(on[t]))
            mean_off = float(np.mean(off[t]))
            effects.append(mean_on - mean_off)
        plt.plot(ts, effects, marker="o", linewidth=2, label=f"{graph}: semantic(bot on - off)")

    plt.axhline(0.0, color="#374151", linewidth=1.0, linestyle="--")
    plt.xlabel("Timestep")
    plt.ylabel("Variance Difference")
    plt.title("Matrix: Bot Effect Over Time")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_matrix_degroot_semantic_gap(rows: list[dict[str, object]], out_path: Path) -> None:
    series: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        graph = str(row["graph"])
        model = str(row["model"])
        bot = str(row["bot"])
        t = int(row["t"])
        key = (graph, f"{model}|{bot}")
        series[key][t].append(float(row["variance"]))

    graphs = sorted({g for g, _ in series.keys()})
    plt.figure(figsize=(11, 6))
    for graph in graphs:
        sem = series.get((graph, "semantic|off"), {})
        deg = series.get((graph, "degroot|off"), {})
        ts = sorted(set(sem.keys()) & set(deg.keys()))
        if not ts:
            continue
        gaps: list[float] = []
        for t in ts:
            mean_sem = float(np.mean(sem[t]))
            mean_deg = float(np.mean(deg[t]))
            gaps.append(mean_sem - mean_deg)
        plt.plot(ts, gaps, marker="o", linewidth=2, label=f"{graph}: semantic(off) - degroot(off)")

    plt.xlabel("Timestep")
    plt.ylabel("Variance Gap")
    plt.title("Matrix: Semantic vs DeGroot Gap Over Time")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _side_entropy(left: int, center_left: int, center_right: int, right: int) -> float:
    counts = np.array([left, center_left, center_right, right], dtype=float)
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    probs = counts / total
    nonzero = probs[probs > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def _plot_matrix_side_entropy(rows: list[dict[str, object]], out_path: Path) -> None:
    by_condition_t: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if str(row["model"]) != "semantic":
            continue
        if row["left_count"] is None:
            continue
        graph = str(row["graph"])
        bot = str(row["bot"])
        t = int(row["t"])
        entropy = _side_entropy(
            int(row["left_count"]),
            int(row["center_left_count"]),
            int(row["center_right_count"]),
            int(row["right_count"]),
        )
        by_condition_t[(graph, bot)][t].append(entropy)

    plt.figure(figsize=(11, 6))
    for cond in sorted(by_condition_t.keys()):
        ts = sorted(by_condition_t[cond].keys())
        means = [float(np.mean(by_condition_t[cond][t])) for t in ts]
        label = f"{cond[0]} | semantic | bot={cond[1]}"
        plt.plot(ts, means, marker="o", linewidth=2, label=label)

    plt.xlabel("Timestep")
    plt.ylabel("Side Entropy (bits)")
    plt.title("Matrix: Semantic Side-Mix Entropy")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _plot_matrix_analysis_pack(rows: list[dict[str, object]], out_dir: Path, final_t: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [
        out_dir / "final_step_variance_bars.png",
        out_dir / "variance_heatmap.png",
        out_dir / "bot_effect_over_time.png",
        out_dir / "semantic_degroot_gap.png",
        out_dir / "semantic_side_entropy.png",
    ]
    _plot_matrix_final_step_bars(rows, files[0], final_t=final_t)
    _plot_matrix_variance_heatmap(rows, files[1])
    _plot_matrix_bot_effect(rows, files[2])
    _plot_matrix_degroot_semantic_gap(rows, files[3])
    _plot_matrix_side_entropy(rows, files[4])
    return files


def main_matrix(args: argparse.Namespace) -> dict[str, object]:
    from src.intervention import add_bot, run_with_bot_on_graph
    from src.simulation import create_agents, run_semantic

    matrix_id, matrix_dir = _make_experiment_dir(
        "matrix",
        "er-rgglr",
        f"seeds-{len(SEED_LIST)}",
        f"steps-{DEFAULT_STEPS}",
    )
    rows: list[dict[str, object]] = []
    matrix_graphs = ("er", "rgglr")
    matrix_steps = DEFAULT_STEPS
    matrix_topic = DEFAULT_TOPIC
    matrix_bot_prob = DEFAULT_BOT_POST_PROB

    print(
        "Running matrix (canonical config): "
        f"graphs={','.join(matrix_graphs)} | seeds={SEED_LIST} | steps={matrix_steps} | topic={matrix_topic}"
    )
    print(f"Output folder: {matrix_dir}")

    for graph_key in matrix_graphs:
        for seed in SEED_LIST:
            G, graph_label = _build_graph(graph_key, seed)
            if G.number_of_nodes() != DEFAULT_N:
                raise ValueError(f"{graph_label} graph must initialize with {DEFAULT_N} nodes.")

            base_metrics = _graph_structure_metrics(G)
            print(f"\n[{graph_label} seed={seed}] DeGroot baseline...")
            degroot_var = _degroot_variance_series(G, steps=matrix_steps)
            _append_matrix_rows(
                rows,
                matrix_id=matrix_id,
                graph=graph_key,
                model="degroot",
                bot="off",
                seed=seed,
                steps=matrix_steps,
                topic=matrix_topic,
                variances=degroot_var,
                side_counts=None,
                graph_metrics=base_metrics,
                bot_degree=None,
            )

            print(f"[{graph_label} seed={seed}] Semantic (no bot)...")
            semantic_log_path = _matrix_log_path(
                matrix_dir,
                graph_key,
                model="semantic",
                bot="off",
                seed=seed,
                enabled=args.log_runs,
            )
            agents = create_agents(G, topic=matrix_topic)
            semantic_var, semantic_counts = run_semantic(
                G=G,
                agents=agents,
                topic=matrix_topic,
                steps=matrix_steps,
                show_progress=args.show_progress,
                log_path=semantic_log_path,
            )
            _append_matrix_rows(
                rows,
                matrix_id=matrix_id,
                graph=graph_key,
                model="semantic",
                bot="off",
                seed=seed,
                steps=matrix_steps,
                topic=matrix_topic,
                variances=semantic_var,
                side_counts=semantic_counts,
                graph_metrics=base_metrics,
                bot_degree=None,
            )

            print(f"[{graph_label} seed={seed}] Semantic (+ bot)...")
            bot_metric_agents = create_agents(G, topic=matrix_topic)
            G_with_bot, _ = add_bot(G, bot_metric_agents, seed=seed)
            bot_metrics = _graph_structure_metrics(G_with_bot)
            bot_degree = int(G_with_bot.degree(G_with_bot.number_of_nodes() - 1))

            bot_log_path = _matrix_log_path(
                matrix_dir,
                graph_key,
                model="semantic",
                bot="on",
                seed=seed,
                enabled=args.log_runs,
            )
            semantic_bot_var, semantic_bot_counts = run_with_bot_on_graph(
                G=G,
                topic=matrix_topic,
                steps=matrix_steps,
                bot_post_prob=matrix_bot_prob,
                seed=seed,
                log_path=bot_log_path,
                show_progress=args.show_progress,
            )
            _append_matrix_rows(
                rows,
                matrix_id=matrix_id,
                graph=graph_key,
                model="semantic",
                bot="on",
                seed=seed,
                steps=matrix_steps,
                topic=matrix_topic,
                variances=semantic_bot_var,
                side_counts=semantic_bot_counts,
                graph_metrics=bot_metrics,
                bot_degree=bot_degree,
            )

    out_path = matrix_dir / "matrix_results.csv"
    _write_matrix_csv(rows, out_path)

    print(f"\nMatrix complete. Wrote {len(rows)} rows to {out_path}")
    summary = _print_matrix_summary(rows, final_t=matrix_steps)

    condition_plot = matrix_dir / "condition_variance_trajectories.png"
    _plot_matrix_condition_lines(rows, condition_plot)
    print(f"Saved {condition_plot}")

    analysis_plots = _plot_matrix_analysis_pack(rows, matrix_dir, final_t=matrix_steps)
    for p in analysis_plots:
        print(f"Saved {p}")

    summary_path = matrix_dir / "matrix_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "matrix_id": matrix_id,
                "output_dir": str(matrix_dir),
                "graphs": list(matrix_graphs),
                "seeds": SEED_LIST,
                "steps": matrix_steps,
                "topic": matrix_topic,
                "final_step_summary": summary,
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

    p_run = sub.add_parser("run", help="Run one canonical condition")
    p_run.add_argument("--graph", choices=["er", "rgglr"], required=True)
    p_run.add_argument("--bot", choices=["off", "on"], required=True)
    p_run.add_argument("--model", choices=["semantic", "degroot", "both"], default="both")
    p_run.add_argument("--seed", type=int, choices=SEED_LIST, default=DEFAULT_SEED)
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
