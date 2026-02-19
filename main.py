#!/usr/bin/env python3
"""
Semantic Opinion Dynamics: LLM Agents on Complex Networks

Main entry point for running the simulation, measurement, and intervention study.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import matplotlib.pyplot as plt

from src.config import DEFAULT_TOPIC, DEFAULT_STEPS
from src.network import create_graph


def main_semantic(args):
    """Run semantic simulation (LLM-based opinion dynamics)."""
    from src.simulation import create_agents, run_semantic
    from src.measurement import embed_opinions, semantic_variance

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
        plt.figure()
        plt.plot(variances, marker="o")
        plt.xlabel("Timestep")
        plt.ylabel("Semantic Variance")
        plt.title(f"Semantic Opinion Dynamics (SBM)\nTopic: {args.topic}")
        plt.grid(True)
        plt.savefig("semantic_variance.png", dpi=150)
        print("\nSaved semantic_variance.png")
    return variances


def main_degroot(args):
    """Run DeGroot consensus for comparison (scalar opinions)."""
    from src.network import run_degroot

    print("Creating network (SBM: 20 nodes, 4 blocks)...")
    G = create_graph(seed=args.seed)
    n = G.number_of_nodes()

    # Random initial scalar opinions in [0, 1]
    rng = np.random.default_rng(args.seed)
    initial = rng.uniform(0, 1, n)

    history = run_degroot(G, initial, steps=args.steps)

    # Classical variance (for scalar opinions)
    variances = [float(np.var(h)) for h in history]
    print("\nDeGroot opinion variance over time:")
    for t, v in enumerate(variances):
        print(f"  t={t}: {v:.4f}")

    if args.plot:
        plt.figure()
        plt.plot(variances, marker="s", color="orange")
        plt.xlabel("Timestep")
        plt.ylabel("Opinion Variance")
        plt.title("DeGroot Consensus (SBM)")
        plt.grid(True)
        plt.savefig("degroot_variance.png", dpi=150)
        print("\nSaved degroot_variance.png")
    return variances


def main_intervention(args):
    """Run intervention study (disinformation bot on SBM)."""
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
        plt.figure()
        plt.plot(variances, marker="o")
        plt.xlabel("Timestep")
        plt.ylabel("Semantic Variance")
        plt.title("SBM: Resilience to Disinformation Bot")
        plt.grid(True)
        plt.savefig("intervention_comparison.png", dpi=150)
        print("\nSaved intervention_comparison.png")
    return variances


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Opinion Dynamics: LLM Agents on Complex Networks"
    )
    sub = parser.add_subparsers(dest="mode", help="Mode to run")

    # Semantic mode
    p_sem = sub.add_parser("semantic", help="Run semantic (LLM) simulation")
    p_sem.add_argument("--topic", default=DEFAULT_TOPIC)
    p_sem.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p_sem.add_argument("--seed", type=int, default=42)
    p_sem.add_argument("--plot", action="store_true", help="Save plots")
    p_sem.set_defaults(func=main_semantic)

    # DeGroot mode
    p_deg = sub.add_parser("degroot", help="Run DeGroot consensus")
    p_deg.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p_deg.add_argument("--seed", type=int, default=42)
    p_deg.add_argument("--plot", action="store_true")
    p_deg.set_defaults(func=main_degroot)

    # Intervention mode
    p_int = sub.add_parser("intervention", help="Run disinformation bot intervention study")
    p_int.add_argument("--topic", default=DEFAULT_TOPIC)
    p_int.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p_int.add_argument("--bot-prob", type=float, default=0.8)
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
