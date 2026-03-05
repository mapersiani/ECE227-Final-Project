#!/usr/bin/env python3
"""
Single-entry CLI for final experiments.

Run shape:
    python main.py run --graph {er|rgglr} --bot {off|on} --seed 42
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import (
    DEGROOT_SCALAR_BY_BLOCK,
    DEFAULT_N,
    DEFAULT_SEED,
    DEFAULT_STEPS,
    DEFAULT_TOPIC,
    LONG_RANGE_FRACTION,
    LONG_RANGE_K,
    PERSONA_BLOCKS,
    RGG_RADIUS,
)

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _save_plot(filename: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR / filename


def _make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _degroot_variance_series(G, steps: int) -> list[float]:
    from src.network import run_degroot

    side_map = DEGROOT_SCALAR_BY_BLOCK
    n = G.number_of_nodes()
    sides = [G.nodes[i].get("side", "center_left") for i in range(n)]
    initial_scalar = np.array([side_map.get(s, 0.5) for s in sides], dtype=float)
    history = run_degroot(G, initial_scalar, steps=steps)
    return [float(np.var(h)) for h in history]


def main_run(args: argparse.Namespace) -> dict[str, list[float]]:
    from src.graphs.er import create_er_graph
    from src.graphs.rgg_long_range import RGGLongRangeParams, create_rgg_long_range_graph
    from src.intervention import run_with_bot_on_graph
    from src.network import load_nodes
    from src.simulation import create_agents, run_semantic

    if args.graph == "er":
        G = create_er_graph(edge_prob=args.edge_prob, seed=args.seed)
        graph_label = "ER"
    else:
        nodes = load_nodes()
        params = RGGLongRangeParams(
            radius=args.radius,
            long_range_fraction=args.long_range_fraction,
            long_range_k=args.long_range_k,
            seed=args.seed,
        )
        G = create_rgg_long_range_graph(nodes, params)
        graph_label = "RGGLR"

    if G.number_of_nodes() != DEFAULT_N:
        raise ValueError(f"{graph_label} graph must initialize with {DEFAULT_N} nodes.")

    if args.bot == "on" and args.model in {"degroot", "both"}:
        raise ValueError("DeGroot comparison currently supports --bot off only.")

    run_id = f"{graph_label}_{'bot' if args.bot == 'on' else 'no_bot'}"
    print(f"Running {run_id} | model={args.model} | topic={DEFAULT_TOPIC}")
    print(f"Graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, seed={args.seed}")

    log_path = None
    if not args.no_log:
        ts = _make_run_id()
        log_path = OUTPUT_DIR / "logs" / f"run_{run_id}_{ts}_seed{args.seed}.jsonl"
        print(f"Logging interactions to {log_path}")

    semantic_var: list[float] | None = None
    degroot_var: list[float] | None = None
    side_counts: list[dict[str, int]] | None = None

    if args.model in {"semantic", "both"}:
        if args.bot == "on":
            semantic_var, side_counts = run_with_bot_on_graph(
                G=G,
                topic=DEFAULT_TOPIC,
                steps=args.steps,
                bot_post_prob=args.bot_prob,
                seed=args.seed,
                log_path=log_path,
                show_progress=True,
            )
        else:
            agents = create_agents(G, topic=DEFAULT_TOPIC, seed=args.seed)
            semantic_var, side_counts = run_semantic(
                G=G,
                agents=agents,
                topic=DEFAULT_TOPIC,
                steps=args.steps,
                log_path=log_path,
            )
        print("\nSemantic variance over time:")
        for t, v in enumerate(semantic_var):
            print(f"  t={t}: {v:.4f}")

    if args.model in {"degroot", "both"}:
        degroot_var = _degroot_variance_series(G, steps=args.steps)
        print("\nDeGroot variance over time:")
        for t, v in enumerate(degroot_var):
            print(f"  t={t}: {v:.4f}")

    if args.plot:
        if args.model == "semantic" and semantic_var is not None:
            out = _save_plot(f"{run_id.lower()}_semantic_variance.png")
            plt.figure()
            plt.plot(semantic_var, marker="o")
            plt.xlabel("Timestep")
            plt.ylabel("Semantic Variance")
            plt.title(f"{run_id}: Semantic Variance")
            plt.grid(True)
            plt.savefig(out, dpi=150)
            print(f"\nSaved {out}")
        elif args.model == "degroot" and degroot_var is not None:
            out = _save_plot(f"{run_id.lower()}_degroot_variance.png")
            plt.figure()
            plt.plot(degroot_var, marker="s", color="orange")
            plt.xlabel("Timestep")
            plt.ylabel("Opinion Variance")
            plt.title(f"{run_id}: DeGroot Variance")
            plt.grid(True)
            plt.savefig(out, dpi=150)
            print(f"\nSaved {out}")
        elif semantic_var is not None and degroot_var is not None:
            out = _save_plot(f"{run_id.lower()}_semantic_vs_degroot.png")
            plt.figure()
            plt.plot(semantic_var, marker="o", label="Semantic (LLM)")
            plt.plot(degroot_var, marker="s", label="DeGroot")
            plt.xlabel("Timestep")
            plt.ylabel("Variance")
            plt.title(f"{run_id}: Semantic vs DeGroot")
            plt.legend()
            plt.grid(True)
            plt.savefig(out, dpi=150)
            print(f"\nSaved {out}")

        if side_counts is not None:
            out2 = _save_plot(f"{run_id.lower()}_side_counts.png")
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
            plt.savefig(out2, dpi=150)
            print(f"Saved {out2}")

    out: dict[str, list[float]] = {}
    if semantic_var is not None:
        out["semantic"] = semantic_var
    if degroot_var is not None:
        out["degroot"] = degroot_var
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one final experiment condition: graph {ER, RGGLR} x bot {off, on}."
    )
    sub = parser.add_subparsers(dest="mode", help="Only supported mode: run")

    p_run = sub.add_parser("run", help="Run one canonical condition")
    p_run.add_argument("--graph", choices=["er", "rgglr"], required=True)
    p_run.add_argument("--bot", choices=["off", "on"], required=True)
    p_run.add_argument("--model", choices=["semantic", "degroot", "both"], default="both")
    p_run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p_run.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p_run.add_argument("--bot-prob", type=float, default=0.8, help="Bot posting amplification probability")
    p_run.add_argument("--edge-prob", type=float, default=0.15, help="ER edge probability (graph=er)")
    p_run.add_argument("--radius", type=float, default=RGG_RADIUS, help="RGG radius (graph=rgglr)")
    p_run.add_argument(
        "--long-range-fraction",
        type=float,
        default=LONG_RANGE_FRACTION,
        help="Fraction of nodes receiving long-range links (graph=rgglr)",
    )
    p_run.add_argument("--long-range-k", type=int, default=LONG_RANGE_K, help="Long-range links per selected node")
    p_run.add_argument("--plot", action="store_true")
    p_run.add_argument("--no-log", action="store_true", help="Disable writing outputs/logs/*.jsonl interaction log")
    p_run.set_defaults(func=main_run)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.mode != "run":
        parser.print_help()
        return 1
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
