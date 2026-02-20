#!/usr/bin/env python3
"""
Semantic Opinion Dynamics: LLM Agents on Complex Networks

CLI entry point. Run semantic (LLM-based) and/or DeGroot simulations, compare them,
or run the disinformation bot intervention study. Plots saved to outputs/ with --plot.

Usage:
    python main.py semantic --steps 5 --plot
    python main.py degroot --steps 5 --plot
    python main.py compare --steps 5 --plot
    python main.py intervention --steps 5 --plot
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "outputs"

import numpy as np
import matplotlib.pyplot as plt

from src.config import DEFAULT_TOPIC, DEFAULT_STEPS
from src.network import create_graph


def _save_plot(filename: str) -> Path:
    """Create outputs/ if needed and return path for saving plot."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR / filename


def main_semantic(args):
    """Run semantic simulation only. Agents update opinions via LLM (Ollama). ~5–10 min for 5 steps."""
    from src.simulation import create_agents, run_semantic

    print("Creating network (SBM: 20 nodes, 4 blocks)...")
    G = create_graph(seed=args.seed)
    print(f"Graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    print("Creating agents...")
    agents = create_agents(G, topic=args.topic, seed=args.seed)
    print("Running semantic simulation...")
    variances = run_semantic(G, agents, args.topic, steps=args.steps)

    print("\nSemantic variance over time:")
    for t, v in enumerate(variances):
        print(f"  t={t}: {v:.4f}")

    if args.plot:
        out = _save_plot("semantic_variance.png")
        plt.figure()
        plt.plot(variances, marker="o")
        plt.xlabel("Timestep")
        plt.ylabel("Semantic Variance")
        plt.title(f"Semantic Opinion Dynamics (SBM)\nTopic: {args.topic}")
        plt.grid(True)
        plt.savefig(out, dpi=150)
        print(f"\nSaved {out}")
    return variances


def main_degroot(args):
    """Run DeGroot consensus only. Fast (seconds). Scalar opinions average toward consensus."""
    from src.network import run_degroot

    print("Creating network (SBM: 20 nodes, 4 blocks)...")
    G = create_graph(seed=args.seed)
    n = G.number_of_nodes()
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
        plt.title("DeGroot Consensus (SBM)")
        plt.grid(True)
        plt.savefig(out, dpi=150)
        print(f"\nSaved {out}")
    return variances


def main_compare(args):
    """Run DeGroot (fast) then semantic (slow). Plot both on one figure for comparison."""
    from src.network import run_degroot
    from src.simulation import create_agents, run_semantic

    print("Creating network (SBM: 20 nodes, 4 blocks)...")
    G = create_graph(seed=args.seed)
    n = G.number_of_nodes()
    blocks = {i: G.nodes[i].get("block", 0) for i in range(n)}
    initial_scalar = np.array([(blocks[i] + 0.5) / 4 for i in range(n)])

    print("Running DeGroot... done (fast)")
    degroot_history = run_degroot(G, initial_scalar, steps=args.steps)
    degroot_var = [float(np.var(h)) for h in degroot_history]
    print("\nDeGroot variance over time:")
    for t, v in enumerate(degroot_var):
        print(f"  t={t}: {v:.4f}")

    print("\nRunning semantic simulation (this takes a few minutes)...")
    agents = create_agents(G, topic=args.topic, seed=args.seed)
    semantic_var = run_semantic(G, agents, args.topic, steps=args.steps)
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
        ax.set_title(f"Semantic vs DeGroot Opinion Dynamics (SBM)\nTopic: {args.topic}")
        ax.legend()
        ax.grid(True)
        plt.savefig(out, dpi=150)
        print(f"\nSaved {out}")
    return semantic_var, degroot_var


def main_intervention(args):
    """Run semantic simulation with a disinformation bot. Measures resilience to semantic drift."""
    from src.intervention import run_with_bot

    variances = run_with_bot(
        topic=args.topic,
        steps=args.steps,
        bot_post_prob=args.bot_prob,
        seed=args.seed,
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
        plt.title("SBM: Resilience to Disinformation Bot")
        plt.grid(True)
        plt.savefig(out, dpi=150)
        print(f"\nSaved {out}")
    return variances


def main():
    """Parse CLI and dispatch to mode handler."""
    parser = argparse.ArgumentParser(
        description="Semantic Opinion Dynamics: LLM Agents on Complex Networks"
    )
    sub = parser.add_subparsers(dest="mode", help="Mode to run")

    p_sem = sub.add_parser("semantic", help="Run semantic (LLM) simulation")
    p_sem.add_argument("--topic", default=DEFAULT_TOPIC)
    p_sem.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p_sem.add_argument("--seed", type=int, default=42)
    p_sem.add_argument("--plot", action="store_true", help="Save plot to outputs/")
    p_sem.set_defaults(func=main_semantic)

    p_cmp = sub.add_parser("compare", help="Run semantic and DeGroot, plot comparison")
    p_cmp.add_argument("--topic", default=DEFAULT_TOPIC)
    p_cmp.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p_cmp.add_argument("--seed", type=int, default=42)
    p_cmp.add_argument("--plot", action="store_true", help="Save plot to outputs/")
    p_cmp.set_defaults(func=main_compare)

    p_deg = sub.add_parser("degroot", help="Run DeGroot consensus only")
    p_deg.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p_deg.add_argument("--seed", type=int, default=42)
    p_deg.add_argument("--plot", action="store_true")
    p_deg.set_defaults(func=main_degroot)

    p_int = sub.add_parser("intervention", help="Run disinformation bot intervention study")
    p_int.add_argument("--topic", default=DEFAULT_TOPIC)
    p_int.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p_int.add_argument("--bot-prob", type=float, default=0.8)
    p_int.add_argument("--seed", type=int, default=42)
    p_int.add_argument("--plot", action="store_true")
    p_int.set_defaults(func=main_intervention)

    args = parser.parse_args()
    if args.mode is None:
        parser.print_help()
        return 1
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
