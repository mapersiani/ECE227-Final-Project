#!/usr/bin/env python3
"""
Single-entry CLI for final experiments.

Run one condition:
    python main.py run --graph {er|rgglr} --bot {off|on} --persona-set {personas|senate}

Run full comparison matrix:
    python main.py matrix
"""

from __future__ import annotations

import argparse
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

import numpy as np

from src.config import (
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
from src.analysis import (
    append_matrix_rows,
    accumulate_final_side_transitions,
    accumulate_side_transitions,
    accumulate_transition_timing,
    graph_structure_metrics,
    print_matrix_summary,
    write_graph_structure_summary_csv,
    write_matrix_csv,
    write_run_timeseries_csv,
    write_transition_summary_csv,
    write_vote_summary_csv,
)
from src.visualization import (
    plot_drift_network,
    plot_matrix_analysis_pack,
    plot_matrix_condition_lines,
    plot_side_counts,
    plot_single_series,
    plot_topology,
    plot_vote_comparison,
)

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Run single condition
# ---------------------------------------------------------------------------

def main_run(args: argparse.Namespace) -> dict[str, list[float]]:
    from src.intervention import add_bot, run_with_bot_on_graph
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
    steps = getattr(args, "steps", DEFAULT_STEPS)

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

    run_id = f"{graph_label}_{'bot' if args.bot == 'on' else 'no_bot'}"
    print(f"Running {run_id} | topic={DEFAULT_TOPIC} | persona_set={persona_set} | steps={steps}")
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
    initial_votes = None
    final_votes = None

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

    degroot_history = run_degroot(G, initial_degroot, steps=steps)
    degroot_var = [float(np.var(x)) for x in degroot_history]
    print("DeGroot variance over time:")
    for t, v in enumerate(degroot_var):
        print(f"  t={t}: {v:.4f}")

    # Run semantic
    base_agents = create_agents(G, topic=DEFAULT_TOPIC)
    if args.bot == "on":
        G_with_bot, bot_agents = add_bot(G, base_agents, seed=args.seed)
        semantic_var, semantic_pol, semantic_drift, side_counts, initial_votes, final_votes, semantic_graph, semantic_agents = run_with_bot_on_graph(
            G=G_with_bot,
            agents=bot_agents,
            topic=DEFAULT_TOPIC,
            steps=steps,
            seed=args.seed,
            log_path=log_path,
            show_progress=True,
            return_state=True,
            persona_set=persona_set,
        )
    else:
        semantic_var, semantic_pol, semantic_drift, side_counts, initial_votes, final_votes = run_semantic(
            G=G,
            agents=base_agents,
            topic=DEFAULT_TOPIC,
            steps=steps,
            log_path=log_path,
            persona_set=persona_set,
        )
        semantic_agents = base_agents
        semantic_graph = G
    print("\nSemantic variance over time:")
    for t, v in enumerate(semantic_var):
        print(f"  t={t}: {v:.4f}")

    # --- Plots ---
    display_graph = semantic_graph if args.bot == "on" and semantic_graph is not None else G

    topology_path = _save_plot(run_dir, "network_topology.png")
    plot_topology(display_graph, topology_path, title=f"{run_id}: Network Topology", seed=args.seed)
    print(f"Saved {topology_path}")

    if degroot_var is not None:
        out_deg = _save_plot(run_dir, "degroot_variance.png")
        plot_single_series(degroot_var, out_deg, f"{run_id}: DeGroot Variance", "DeGroot Variance", marker="s", color="orange")
        print(f"Saved {out_deg}")

    if semantic_var is not None:
        out = _save_plot(run_dir, "semantic_variance.png")
        plot_single_series(semantic_var, out, f"{run_id}: Semantic Variance", "Semantic Variance")
        print(f"Saved {out}")

    if semantic_pol is not None:
        out_pol = _save_plot(run_dir, "semantic_polarization.png")
        plot_single_series(semantic_pol, out_pol, f"{run_id}: Opinion Polarization Over Time", "Opinion Polarization", color="purple")
        print(f"Saved {out_pol}")

    if semantic_drift is not None:
        out_drift = _save_plot(run_dir, "persona_drift.png")
        plot_single_series(semantic_drift, out_drift, f"{run_id}: Persona Drift Over Time", "Persona Drift Mean", color="red")
        print(f"Saved {out_drift}")

    if side_counts is not None:
        out2 = _save_plot(run_dir, "side_counts.png")
        plot_side_counts(side_counts, out2, f"{run_id}: Side Counts")
        print(f"Saved {out2}")

    if semantic_agents is not None:
        drift_path = _save_plot(run_dir, "opinion_drift_network.png")
        plot_drift_network(semantic_graph, semantic_agents, drift_path, title=f"{run_id}: Opinion Drift by Node", seed=args.seed)
        print(f"Saved {drift_path}")

    timeseries_path = run_dir / "timeseries.csv"
    write_run_timeseries_csv(timeseries_path, semantic_var, side_counts)
    print(f"Saved {timeseries_path}")

    if semantic_agents is not None and initial_votes is not None and final_votes is not None:
        vote_plot_path = run_dir / "vote_comparison.png"
        plot_vote_comparison(initial_votes, final_votes, vote_plot_path, title=f"{run_id}: Initial vs Final Votes")
        print(f"Saved {vote_plot_path}")

    # --- Run summary ---
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
        "steps": steps,
        "config": {
            "edge_prob": DEFAULT_ER_EDGE_PROB,
            "rgg_radius": RGG_RADIUS,
            "rgg_long_range_fraction": LONG_RANGE_FRACTION,
            "rgg_long_range_k": LONG_RANGE_K,
            "log_mode": DEFAULT_LOG_MODE,
            "max_neighbors_per_update": MAX_NEIGHBORS_PER_UPDATE,
            "max_chars_per_neighbor": MAX_CHARS_PER_NEIGHBOR,
        },
        "graph_metrics": graph_structure_metrics(display_graph),
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


# ---------------------------------------------------------------------------
# Run full matrix
# ---------------------------------------------------------------------------

def main_matrix(args: argparse.Namespace) -> dict[str, object]:
    from src.intervention import add_bot, run_with_bot_on_graph
    from src.simulation import create_agents, run_semantic
    from src.degroot import run_degroot
    from src.config import PERSONA_BLOCK_LAYOUT, side_from_name

    steps = getattr(args, "steps", DEFAULT_STEPS)

    matrix_id, matrix_dir = _make_experiment_dir(
        "matrix",
        "er-rgglr",
        "personas-senate",
        f"seeds-{len(SEED_LIST)}",
        f"steps-{steps}",
    )
    rows: list[dict[str, object]] = []
    matrix_graphs = ("er", "rgglr")
    matrix_persona_sets = ("personas", "senate")
    matrix_topic = DEFAULT_TOPIC

    def _init_transitions() -> dict[str, np.ndarray]:
        return {
            "off": np.zeros((len(PERSONA_BLOCKS), len(PERSONA_BLOCKS)), dtype=float),
            "on": np.zeros((len(PERSONA_BLOCKS), len(PERSONA_BLOCKS)), dtype=float),
        }

    def _init_timing(s: int) -> dict[str, np.ndarray]:
        return {
            "off": np.zeros(s, dtype=float),
            "on": np.zeros(s, dtype=float),
        }

    transitions_by_persona_by_bot: dict[str, dict[str, np.ndarray]] = {
        p: _init_transitions() for p in matrix_persona_sets
    }
    final_transitions_by_persona_by_bot: dict[str, dict[str, np.ndarray]] = {
        p: _init_transitions() for p in matrix_persona_sets
    }
    changed_counts_by_persona_by_bot: dict[str, dict[str, np.ndarray]] = {
        p: _init_timing(steps) for p in matrix_persona_sets
    }
    total_counts_by_persona_by_bot: dict[str, dict[str, np.ndarray]] = {
        p: _init_timing(steps) for p in matrix_persona_sets
    }

    print(
        "Running matrix (canonical config): "
        f"graphs={','.join(matrix_graphs)} | persona_sets={','.join(matrix_persona_sets)} | "
        f"seeds={SEED_LIST} | steps={steps} | topic={matrix_topic}"
    )
    print(f"Output folder: {matrix_dir}")

    rep_seed = SEED_LIST[0]
    for graph_key in matrix_graphs:
        for persona_set in matrix_persona_sets:
            G_top, graph_label = _build_graph(graph_key, rep_seed, persona_set=persona_set)
            topo_path = matrix_dir / f"network_topology_{graph_key}_{persona_set}.png"
            plot_topology(G_top, topo_path, title=f"Matrix: {graph_label} | {persona_set} (seed={rep_seed})", seed=rep_seed)
            print(f"Saved {topo_path}")

    for graph_key in matrix_graphs:
        for persona_set in matrix_persona_sets:
            for seed in SEED_LIST:
                G, graph_label = _build_graph(graph_key, seed, persona_set=persona_set)
                base_metrics = graph_structure_metrics(G)

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

                degroot_history = run_degroot(G, initial_degroot, steps=steps)
                degroot_var = [float(np.var(x)) for x in degroot_history]

                append_matrix_rows(
                    rows,
                    matrix_id=matrix_id,
                    graph=graph_key,
                    persona_set=persona_set,
                    model="degroot",
                    bot="off",
                    seed=seed,
                    steps=steps,
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
                    matrix_dir, graph_key, persona_set, model="semantic", bot="off", seed=seed, enabled=args.log_runs,
                )
                agents = create_agents(G, topic=matrix_topic)
                semantic_var, semantic_pol, semantic_drift, semantic_counts, semantic_labels, initial_votes, final_votes = run_semantic(
                    G=G,
                    agents=agents,
                    topic=matrix_topic,
                    steps=steps,
                    show_progress=args.show_progress,
                    log_path=semantic_log_path,
                    return_side_labels=True,
                    persona_set=persona_set,
                )
                accumulate_side_transitions(semantic_labels, transitions_by_persona_by_bot[persona_set]["off"])
                accumulate_final_side_transitions(semantic_labels, final_transitions_by_persona_by_bot[persona_set]["off"])
                accumulate_transition_timing(
                    semantic_labels,
                    changed_counts_by_persona_by_bot[persona_set]["off"],
                    total_counts_by_persona_by_bot[persona_set]["off"],
                )
                append_matrix_rows(
                    rows,
                    matrix_id=matrix_id,
                    graph=graph_key,
                    persona_set=persona_set,
                    model="semantic",
                    bot="off",
                    seed=seed,
                    steps=steps,
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
                plot_drift_network(
                    G, agents, drift_path_off,
                    title=f"Drift: {graph_label} | {persona_set} (seed={seed}) | Bot: off",
                    seed=seed,
                )

                print(f"[{graph_label} {persona_set} seed={seed}] Semantic (+ bot)...")
                bot_metric_agents = create_agents(G, topic=matrix_topic)
                G_with_bot, bot_agents = add_bot(G, bot_metric_agents, seed=seed)
                bot_metrics = graph_structure_metrics(G_with_bot)
                bot_degree = int(G_with_bot.degree(G_with_bot.number_of_nodes() - 1))

                bot_log_path = _matrix_log_path(
                    matrix_dir, graph_key, persona_set, model="semantic", bot="on", seed=seed, enabled=args.log_runs,
                )
                semantic_bot_var, semantic_bot_pol, semantic_bot_drift, semantic_bot_counts, semantic_bot_labels, initial_votes_bot, final_votes_bot, G_bot_ret, bot_agents_ret = run_with_bot_on_graph(
                    G=G_with_bot,
                    agents=bot_agents,
                    topic=matrix_topic,
                    steps=steps,
                    seed=seed,
                    log_path=bot_log_path,
                    show_progress=args.show_progress,
                    return_side_labels=True,
                    return_state=True,
                    persona_set=persona_set,
                )
                accumulate_side_transitions(semantic_bot_labels, transitions_by_persona_by_bot[persona_set]["on"])
                accumulate_final_side_transitions(semantic_bot_labels, final_transitions_by_persona_by_bot[persona_set]["on"])
                accumulate_transition_timing(
                    semantic_bot_labels,
                    changed_counts_by_persona_by_bot[persona_set]["on"],
                    total_counts_by_persona_by_bot[persona_set]["on"],
                )
                append_matrix_rows(
                    rows,
                    matrix_id=matrix_id,
                    graph=graph_key,
                    persona_set=persona_set,
                    model="semantic",
                    bot="on",
                    seed=seed,
                    steps=steps,
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
                plot_drift_network(
                    G_bot_ret, bot_agents_ret, drift_path_on,
                    title=f"Drift: {graph_label} | {persona_set} (seed={seed}) | Bot: on",
                    seed=seed,
                )

    out_path = matrix_dir / "matrix_results.csv"
    write_matrix_csv(rows, out_path)

    print(f"\nMatrix complete. Wrote {len(rows)} rows to {out_path}")
    summary = print_matrix_summary(rows, final_t=steps)

    condition_plot = matrix_dir / "condition_variance_trajectories.png"
    plot_matrix_condition_lines(rows, condition_plot)
    print(f"Saved {condition_plot}")

    analysis_plots = plot_matrix_analysis_pack(
        rows,
        matrix_dir,
        final_t=steps,
        transitions_by_persona_by_bot=transitions_by_persona_by_bot,
        final_transitions_by_persona_by_bot=final_transitions_by_persona_by_bot,
        changed_counts_by_persona_by_bot=changed_counts_by_persona_by_bot,
        total_counts_by_persona_by_bot=total_counts_by_persona_by_bot,
    )
    for p in analysis_plots:
        print(f"Saved {p}")

    transition_summary_path = matrix_dir / "side_transition_summary.csv"
    write_transition_summary_csv(
        transition_summary_path,
        transitions_by_persona_by_bot=transitions_by_persona_by_bot,
        final_transitions_by_persona_by_bot=final_transitions_by_persona_by_bot,
        changed_counts_by_persona_by_bot=changed_counts_by_persona_by_bot,
        total_counts_by_persona_by_bot=total_counts_by_persona_by_bot,
    )
    print(f"Saved {transition_summary_path}")

    vote_summary_path = matrix_dir / "vote_summary.csv"
    write_vote_summary_csv(vote_summary_path, rows, final_t=steps)
    print(f"Saved {vote_summary_path}")

    graph_summary_path = matrix_dir / "graph_structure_summary.csv"
    write_graph_structure_summary_csv(graph_summary_path, rows)
    print(f"Saved {graph_summary_path}")

    summary_path = matrix_dir / "matrix_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "matrix_id": matrix_id,
                "output_dir": str(matrix_dir),
                "graphs": list(matrix_graphs),
                "persona_sets": list(matrix_persona_sets),
                "seeds": SEED_LIST,
                "steps": steps,
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Number of simulation steps (default: {DEFAULT_STEPS})",
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
    p_matrix.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Number of simulation steps (default: {DEFAULT_STEPS})",
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
