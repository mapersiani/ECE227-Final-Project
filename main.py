#!/usr/bin/env python3
"""
Semantic Opinion Dynamics: LLM Agents on ER Graph + DeGroot Baseline.

CLI entry point. Builds an Erdős–Rényi graph on personas from ``nodes.json`` and can:
- Run semantic (LLM-based) opinion dynamics
- Run DeGroot scalar baseline
- Compare both in one plot
- Run a disinformation bot intervention study
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.config import DEFAULT_STEPS, DEFAULT_TOPIC
from src.network import create_graph, export_gephi, run_degroot

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _save_plot(filename: str) -> Path:
    """Create outputs/ if needed and return path for saving plot."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR / filename


def _make_run_id() -> str:
    """Timestamp-based run id used for logs/exports."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main_semantic(args: argparse.Namespace) -> list[float]:
    """Run semantic simulation only. Agents update opinions via LLM (Ollama)."""
    from src.simulation import create_agents, run_semantic

    print("Creating ER network on personas from nodes.json...")
    G = create_graph(edge_prob=args.edge_prob, seed=args.seed)
    print(f"Graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    if args.export_gephi:
        out_dir = OUTPUT_DIR / "gephi"
        base = f"er_p{args.edge_prob}_seed{args.seed}"
        gexf_path, graphml_path = export_gephi(G, out_dir, base)
        print(f"Exported Gephi files: {gexf_path.name}, {graphml_path.name}")
    print("Creating agents...")
    agents = create_agents(G, topic=args.topic, seed=args.seed)
    print("Running semantic simulation (this takes a few minutes)...")
    log_path = None
    if not args.no_log:
        run_id = _make_run_id()
        log_path = OUTPUT_DIR / "logs" / f"semantic_{run_id}_p{args.edge_prob}_seed{args.seed}.jsonl"
        print(f"Logging interactions to {log_path}")
    variances, side_counts = run_semantic(G, agents, args.topic, steps=args.steps, log_path=log_path)

    print("\nSemantic variance over time:")
    for t, v in enumerate(variances):
        print(f"  t={t}: {v:.4f}")

    if args.plot:
        out = _save_plot("semantic_variance.png")
        plt.figure()
        plt.plot(variances, marker="o")
        plt.xlabel("Timestep")
        plt.ylabel("Semantic Variance")
        plt.title(f"Semantic Opinion Dynamics (ER, personas)\nTopic: {args.topic}")
        plt.grid(True)
        plt.savefig(out, dpi=150)
        print(f"\nSaved {out}")
        # Persona-side counts over time
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
    """
    Run DeGroot consensus on an ER graph built on personas from nodes.json.

    Initial opinions are random scalars in [0, 1]; over time they converge toward consensus.
    """
    print("Creating ER network on personas from nodes.json...")
    G = create_graph(edge_prob=args.edge_prob, seed=args.seed)
    n = G.number_of_nodes()
    print(f"Graph: nodes={n}, edges={G.number_of_edges()}")
    if args.export_gephi:
        out_dir = OUTPUT_DIR / "gephi"
        base = f"er_p{args.edge_prob}_seed{args.seed}"
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


def main_compare(args: argparse.Namespace) -> tuple[list[float], list[float]]:
    """Run DeGroot (fast) then semantic (slow). Plot both on one figure for comparison."""
    from src.simulation import create_agents, run_semantic

    print("Creating ER network on personas from nodes.json...")
    G = create_graph(edge_prob=args.edge_prob, seed=args.seed)
    n = G.number_of_nodes()
    print(f"Graph: nodes={n}, edges={G.number_of_edges()}")
    if args.export_gephi:
        out_dir = OUTPUT_DIR / "gephi"
        base = f"er_p{args.edge_prob}_seed{args.seed}"
        gexf_path, graphml_path = export_gephi(G, out_dir, base)
        print(f"Exported Gephi files: {gexf_path.name}, {graphml_path.name}")

    # DeGroot: map ideological side to scalar in [0,1] for a structured baseline
    side_map = {"left": 0.0, "center_left": 1 / 3, "center_right": 2 / 3, "right": 1.0}
    sides = [G.nodes[i].get("side", "center_left") for i in range(n)]
    initial_scalar = np.array([side_map.get(s, 0.5) for s in sides])

    print("Running DeGroot... done (fast)")
    history = run_degroot(G, initial_scalar, steps=args.steps)
    degroot_var = [float(np.var(h)) for h in history]
    print("\nDeGroot variance over time:")
    for t, v in enumerate(degroot_var):
        print(f"  t={t}: {v:.4f}")

    print("\nRunning semantic simulation (this takes a few minutes)...")
    agents = create_agents(G, topic=args.topic, seed=args.seed)
    log_path = None
    if not args.no_log:
        run_id = _make_run_id()
        log_path = OUTPUT_DIR / "logs" / f"compare_{run_id}_p{args.edge_prob}_seed{args.seed}.jsonl"
        print(f"Logging interactions to {log_path}")
    semantic_var, side_counts = run_semantic(G, agents, args.topic, steps=args.steps, log_path=log_path)
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
        ax.set_title(f"Semantic vs DeGroot (ER, personas)\nTopic: {args.topic}")
        ax.legend()
        ax.grid(True)
        plt.savefig(out, dpi=150)
        print(f"\nSaved {out}")
        # Persona-side counts over time for semantic run
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


def main_intervention(args: argparse.Namespace) -> list[float]:
    """Run semantic simulation with a disinformation bot. Measures resilience to semantic drift."""
    from src.intervention import run_with_bot

    log_path = None
    if not args.no_log:
        run_id = _make_run_id()
        log_path = OUTPUT_DIR / "logs" / f"intervention_{run_id}_p{args.edge_prob}_seed{args.seed}.jsonl"
        print(f"Logging interactions to {log_path}")
    variances, side_counts = run_with_bot(
        topic=args.topic,
        steps=args.steps,
        bot_post_prob=args.bot_prob,
        seed=args.seed,
        edge_prob=args.edge_prob,
        log_path=log_path,
    )
    print("\nSemantic variance with disinformation bot:")
    for t, v in enumerate(variances):
        print(f"  t={t}: {v:.4f}")

    if args.plot:
        out = _save_plot("intervention_comparison.png")
        plt.figure()
        plt.plot(variances, marker="o")
        plt.xlabel("Timestep")
        plt.ylabel("Semantic Variance")
        plt.title("ER personas: Resilience to Disinformation Bot")
        plt.grid(True)
        plt.savefig(out, dpi=150)
        print(f"\nSaved {out}")
        # Persona-side counts over time
        out2 = _save_plot("intervention_side_counts.png")
        steps = list(range(len(side_counts)))
        labels = sorted(side_counts[0].keys())
        plt.figure()
        for lab in labels:
            series = [c[lab] for c in side_counts]
            plt.plot(steps, series, marker="o", label=lab)
        plt.xlabel("Timestep")
        plt.ylabel("Count of agents")
        plt.title("Intervention: agents classified by coarse persona side")
        plt.legend()
        plt.grid(True)
        plt.savefig(out2, dpi=150)
        print(f"Saved {out2}")
    return variances


def main() -> int:
    """Parse CLI arguments and dispatch to the selected mode."""
    parser = argparse.ArgumentParser(
        description="Semantic opinion dynamics and DeGroot baseline on ER graph of personas."
    )
    sub = parser.add_subparsers(dest="mode", help="Mode to run")

    # Semantic mode
    p_sem = sub.add_parser("semantic", help="Run semantic (LLM) simulation")
    p_sem.add_argument("--topic", default=DEFAULT_TOPIC)
    p_sem.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p_sem.add_argument("--seed", type=int, default=42)
    p_sem.add_argument("--edge-prob", type=float, default=0.15)
    p_sem.add_argument("--plot", action="store_true", help="Save semantic_variance.png")
    p_sem.add_argument("--export-gephi", action="store_true", help="Export ER graph to outputs/gephi/")
    p_sem.add_argument("--no-log", action="store_true", help="Disable writing outputs/logs/*.jsonl interaction log")
    p_sem.set_defaults(func=main_semantic)

    # DeGroot mode
    p_deg = sub.add_parser("degroot", help="Run DeGroot consensus only")
    p_deg.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p_deg.add_argument("--seed", type=int, default=42)
    p_deg.add_argument("--edge-prob", type=float, default=0.15)
    p_deg.add_argument("--plot", action="store_true", help="Save degroot_variance.png")
    p_deg.add_argument("--export-gephi", action="store_true", help="Export ER graph to outputs/gephi/")
    p_deg.set_defaults(func=main_degroot)

    # Compare mode
    p_cmp = sub.add_parser("compare", help="Run semantic and DeGroot, plot comparison")
    p_cmp.add_argument("--topic", default=DEFAULT_TOPIC)
    p_cmp.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p_cmp.add_argument("--seed", type=int, default=42)
    p_cmp.add_argument("--edge-prob", type=float, default=0.15)
    p_cmp.add_argument("--plot", action="store_true", help="Save semantic_vs_degroot.png")
    p_cmp.add_argument("--export-gephi", action="store_true", help="Export ER graph to outputs/gephi/")
    p_cmp.add_argument("--no-log", action="store_true", help="Disable writing outputs/logs/*.jsonl interaction log")
    p_cmp.set_defaults(func=main_compare)

    # Intervention mode
    p_int = sub.add_parser("intervention", help="Run disinformation bot intervention study")
    p_int.add_argument("--topic", default=DEFAULT_TOPIC)
    p_int.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p_int.add_argument("--bot-prob", type=float, default=0.8)
    p_int.add_argument("--seed", type=int, default=42)
    p_int.add_argument("--edge-prob", type=float, default=0.15)
    p_int.add_argument("--plot", action="store_true", help="Save intervention_comparison.png")
    p_int.add_argument("--export-gephi", action="store_true", help="Export ER graph to outputs/gephi/")
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
