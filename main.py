#!/usr/bin/env python3
"""
Semantic Opinion Dynamics: LLM Agents on ER Graph + DeGroot Baseline.

CLI entry point. Builds an Erdős–Rényi graph on personas from ``nodes.json`` and can:
- Run semantic (LLM-based) opinion dynamics
- Run DeGroot scalar baseline
- Compare both in one plot
- Run a disinformation bot intervention study

Upgrades:
- ER specified by expected average degree (more comparable than raw p)
- Expose neighbor sampling + truncation knobs for LLM stability
- Pass run_seed through to simulation/intervention for full reproducibility
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import DEFAULT_STEPS, DEFAULT_TOPIC
from src.network import create_graph, export_gephi, run_degroot, graph_stats

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "outputs"

def _aggregate_runs(all_runs):
    """
    all_runs: list of list[float] with equal length.
    returns: mean, std as np arrays
    """
    arr = np.array(all_runs, dtype=float)  # shape (S, T)
    return arr.mean(axis=0), arr.std(axis=0)

def _plot_mean_std(x, mean, std, label):
    plt.plot(x, mean, label=label)
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)


def _save_plot(filename: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR / filename


def _make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_graph_from_args(args: argparse.Namespace):
    """
    Build ER graph. If --avg-degree is provided, it overrides --edge-prob.

    Always ensures no isolates (so everyone updates in semantic mode).
    """
    G = create_graph(
        edge_prob=args.edge_prob,
        seed=args.seed,
        avg_degree=args.avg_degree,
        ensure_no_isolates=True,
    )
    return G


def _base_name(args: argparse.Namespace) -> str:
    if args.avg_degree is not None:
        return f"er_k{args.avg_degree}_seed{args.seed}"
    return f"er_p{args.edge_prob}_seed{args.seed}"


def main_semantic(args: argparse.Namespace) -> list[float]:
    from src.simulation import create_agents, run_semantic

    print("Creating ER network on personas from nodes.json...")
    G = _build_graph_from_args(args)
    st = graph_stats(G)
    print(f"Graph: nodes={st['nodes']}, edges={st['edges']}, isolates={st['isolates']}, "
          f"deg(min/mean/max)={st['min_deg']}/{st['mean_deg']:.2f}/{st['max_deg']}")

    if args.export_gephi:
        out_dir = OUTPUT_DIR / "gephi"
        base = _base_name(args)
        gexf_path, graphml_path = export_gephi(G, out_dir, base)
        print(f"Exported Gephi files: {gexf_path.name}, {graphml_path.name}")

    print("Creating agents...")
    agents = create_agents(G, topic=DEFAULT_TOPIC, seed=args.seed)

    print("Running semantic simulation (LLM)...")
    log_path = None
    if not args.no_log:
        run_id = _make_run_id()
        base = _base_name(args)
        log_path = OUTPUT_DIR / "logs" / f"semantic_{run_id}_{base}.jsonl"
        print(f"Logging interactions to {log_path}")

    variances, side_counts = run_semantic(
        G,
        agents,
        DEFAULT_TOPIC,
        steps=args.steps,
        log_path=log_path,
        run_seed=args.seed,
        max_neighbors=args.max_neighbors,
        max_chars=args.max_chars,
    )

    print("\nSemantic variance over time:")
    for t, v in enumerate(variances):
        print(f"  t={t}: {v:.4f}")

    if args.plot:
        out = _save_plot("semantic_variance.png")
        plt.figure()
        plt.plot(variances, marker="o")
        plt.xlabel("Timestep")
        plt.ylabel("Semantic Variance")
        plt.title(f"Semantic Opinion Dynamics (ER, personas)\nTopic: {DEFAULT_TOPIC}")
        plt.grid(True)
        plt.savefig(out, dpi=150)
        print(f"\nSaved {out}")

        out2 = _save_plot("semantic_side_counts.png")
        steps = list(range(len(side_counts)))
        labels = sorted(side_counts[0].keys())
        plt.figure()
        for lab in labels:
            series = [c[lab] for c in side_counts]
            plt.plot(steps, series, marker="o", label=lab)
        plt.xlabel("Timestep")
        plt.ylabel("Count of agents")
        plt.title("Semantic: agents classified by coarse persona side")
        plt.legend()
        plt.grid(True)
        plt.savefig(out2, dpi=150)
        print(f"Saved {out2}")

    return variances


def main_degroot(args: argparse.Namespace) -> list[float]:
    print("Creating ER network on personas from nodes.json...")
    G = _build_graph_from_args(args)
    n = G.number_of_nodes()
    st = graph_stats(G)
    print(f"Graph: nodes={n}, edges={st['edges']}, isolates={st['isolates']}")

    if args.export_gephi:
        out_dir = OUTPUT_DIR / "gephi"
        base = _base_name(args)
        gexf_path, graphml_path = export_gephi(G, out_dir, base)
        print(f"Exported Gephi files: {gexf_path.name}, {graphml_path.name}")

    rng = np.random.default_rng(args.seed)
    initial = rng.uniform(0, 1, n)
    history = run_degroot(G, initial, steps=args.steps)
    variances = [float(np.var(h)) for h in history]

    print("\nDeGroot opinion variance over time:")
    for t, v in enumerate(variances):
        print(f"  t={t}: {v:.4f}")

    if args.plot:
        out = _save_plot("degroot_variance.png")
        plt.figure()
        plt.plot(variances, marker="s", color="orange")
        plt.xlabel("Timestep")
        plt.ylabel("Opinion Variance")
        plt.title("DeGroot Consensus on ER Graph (personas)")
        plt.grid(True)
        plt.savefig(out, dpi=150)
        print(f"\nSaved {out}")

    return variances


def main_compare(args: argparse.Namespace):
    from src.simulation import create_agents, run_semantic

    print("Creating ER network on personas from nodes.json...")
    G = _build_graph_from_args(args)
    n = G.number_of_nodes()
    st = graph_stats(G)
    print(f"Graph: nodes={n}, edges={st['edges']}, isolates={st['isolates']}")

    if args.export_gephi:
        out_dir = OUTPUT_DIR / "gephi"
        base = _base_name(args)
        gexf_path, graphml_path = export_gephi(G, out_dir, base)
        print(f"Exported Gephi files: {gexf_path.name}, {graphml_path.name}")

    # DeGroot baseline: map persona side to scalar in [0,1]
    side_map = {"left": 0.0, "center_left": 1 / 3, "center_right": 2 / 3, "right": 1.0}
    sides = [G.nodes[i].get("side", "center_left") for i in range(n)]
    initial_scalar = np.array([side_map.get(s, 0.5) for s in sides])

    print("Running DeGroot... done (fast)")
    history = run_degroot(G, initial_scalar, steps=args.steps)
    degroot_var = [float(np.var(h)) for h in history]

    print("\nDeGroot variance over time:")
    for t, v in enumerate(degroot_var):
        print(f"  t={t}: {v:.4f}")

    print("\nRunning semantic simulation (LLM)...")
    agents = create_agents(G, topic=DEFAULT_TOPIC, seed=args.seed)

    log_path = None
    if not args.no_log:
        run_id = _make_run_id()
        base = _base_name(args)
        log_path = OUTPUT_DIR / "logs" / f"compare_{run_id}_{base}.jsonl"
        print(f"Logging interactions to {log_path}")

    semantic_var, side_counts = run_semantic(
        G,
        agents,
        DEFAULT_TOPIC,
        steps=args.steps,
        log_path=log_path,
        run_seed=args.seed,
        max_neighbors=args.max_neighbors,
        max_chars=args.max_chars,
    )

    print("\nSemantic variance over time:")
    for t, v in enumerate(semantic_var):
        print(f"  t={t}: {v:.4f}")

    if args.plot:
        out = _save_plot("semantic_vs_degroot.png")
        fig, ax = plt.subplots()
        ax.plot(degroot_var, marker="s", color="orange", label="DeGroot (scalar)")
        ax.plot(semantic_var, marker="o", color="steelblue", label="Semantic (LLM)")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Variance")
        ax.set_title(f"Semantic vs DeGroot (ER, personas)\nTopic: {DEFAULT_TOPIC}")
        ax.legend()
        ax.grid(True)
        plt.savefig(out, dpi=150)
        print(f"\nSaved {out}")

        out2 = _save_plot("semantic_side_counts.png")
        steps = list(range(len(side_counts)))
        labels = sorted(side_counts[0].keys())
        plt.figure()
        for lab in labels:
            series = [c[lab] for c in side_counts]
            plt.plot(steps, series, marker="o", label=lab)
        plt.xlabel("Timestep")
        plt.ylabel("Count of agents")
        plt.title("Semantic: agents classified by coarse persona side")
        plt.legend()
        plt.grid(True)
        plt.savefig(out2, dpi=150)
        print(f"Saved {out2}")

    return semantic_var, degroot_var


def main_intervention(args: argparse.Namespace):
    from src.simulation import create_agents, run_semantic
    from src.intervention import run_with_bot

    seeds = args.seeds if args.seeds is not None and len(args.seeds) > 0 else [args.seed]

    all_base = []
    all_bot = []

    for sd in seeds:
        # --- baseline semantic (NO bot) ---
        G = create_graph(
            edge_prob=args.edge_prob,
            seed=sd,
            avg_degree=args.avg_degree,
            ensure_no_isolates=True,
        )
        agents = create_agents(G, topic=DEFAULT_TOPIC, seed=sd)

        base_var, _base_counts = run_semantic(
            G,
            agents,
            DEFAULT_TOPIC,
            steps=args.steps,
            log_path=None,
            run_seed=sd,
            max_neighbors=args.max_neighbors,
            max_chars=args.max_chars,
        )
        all_base.append(base_var)

        # --- semantic WITH bot (uses same seed + same graph spec) ---
        bot_var, _bot_counts = run_with_bot(
            topic=DEFAULT_TOPIC,
            steps=args.steps,
            bot_post_prob=args.bot_prob,
            seed=sd,
            edge_prob=args.edge_prob,
            avg_degree=args.avg_degree,
            log_path=None,
            max_neighbors=args.max_neighbors,
            max_chars=args.max_chars,
        )
        all_bot.append(bot_var)

    base_mean, base_std = _aggregate_runs(all_base)
    bot_mean, bot_std = _aggregate_runs(all_bot)

    print("\nBaseline (no bot) semantic variance mean:")
    for t, v in enumerate(base_mean):
        print(f"  t={t}: {v:.4f}")

    print("\nWith-bot semantic variance mean:")
    for t, v in enumerate(bot_mean):
        print(f"  t={t}: {v:.4f}")

    if args.plot:
        out = _save_plot("intervention_vs_baseline.png")
        plt.figure()
        x = np.arange(len(base_mean))
        _plot_mean_std(x, base_mean, base_std, label="Semantic (no bot)")
        _plot_mean_std(x, bot_mean, bot_std, label=f"Semantic (bot p={args.bot_prob})")
        plt.xlabel("Timestep")
        plt.ylabel("Semantic Variance")
        plt.title("ER personas: baseline vs disinformation bot")
        plt.grid(True)
        plt.legend()
        plt.savefig(out, dpi=150)
        print(f"\nSaved {out}")

    return bot_mean.tolist()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Semantic opinion dynamics and DeGroot baseline on ER graph of personas."
    )
    sub = parser.add_subparsers(dest="mode", help="Mode to run")

    # Common knobs
    def add_common(p):
        p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--edge-prob", type=float, default=0.15)
        p.add_argument("--avg-degree", type=float, default=None, help="If set, overrides --edge-prob with p=k/(n-1)")
        p.add_argument("--max-neighbors", type=int, default=5, help="Max neighbors included in LLM prompt")
        p.add_argument("--max-chars", type=int, default=400, help="Max chars per neighbor opinion in prompt")
        p.add_argument("--plot", action="store_true")
        p.add_argument("--export-gephi", action="store_true", help="Export ER graph to outputs/gephi/")
        p.add_argument("--seeds", type=int, nargs="*", default=None,
               help="Optional list of seeds to average over (e.g., --seeds 0 1 2 3 4). If set, overrides --seed.")

    # Semantic
    p_sem = sub.add_parser("semantic", help="Run semantic (LLM) simulation")
    add_common(p_sem)
    p_sem.add_argument("--no-log", action="store_true", help="Disable writing outputs/logs/*.jsonl interaction log")
    p_sem.set_defaults(func=main_semantic)

    # DeGroot
    p_deg = sub.add_parser("degroot", help="Run DeGroot consensus only")
    add_common(p_deg)
    p_deg.set_defaults(func=main_degroot)

    # Compare
    p_cmp = sub.add_parser("compare", help="Run semantic and DeGroot, plot comparison")
    add_common(p_cmp)
    p_cmp.add_argument("--no-log", action="store_true", help="Disable writing outputs/logs/*.jsonl interaction log")
    p_cmp.set_defaults(func=main_compare)

    # Intervention
    p_int = sub.add_parser("intervention", help="Run disinformation bot intervention study")
    add_common(p_int)
    p_int.add_argument("--bot-prob", type=float, default=0.8)
    p_int.add_argument("--no-log", action="store_true", help="Disable writing outputs/logs/*.jsonl interaction log")
    p_int.set_defaults(func=main_intervention)

    args = parser.parse_args()
    if args.mode is None:
        parser.print_help()
        return 1

    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#python main.py semantic --steps 5 --seed 42 --edge-prob 0.15 --plot
#python main.py degroot --steps 5 --seed 42 --edge-prob 0.15 --plot
#python main.py compare --steps 5 --seed 42 --edge-prob 0.15 --plot
#python main.py intervention --steps 5 --seed 42 --edge-prob 0.15 --bot-prob 0.8 --plot
#or python main.py intervention --steps 5 --seeds 0 1 2 3 4 --avg-degree 6 --plot