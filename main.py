#!/usr/bin/env python3
"""
Semantic Opinion Dynamics: LLM Agents on a Random Geometric Graph with Long-Range Connections

CLI entry point. Run semantic (LLM-based) and/or DeGroot simulations, compare them,
or visualize the RGG topology.

Usage:
    python main.py graph                          # Visualize the RGG (no LLM needed)
    python main.py semantic --steps 5 --plot
    python main.py degroot  --steps 5 --plot
    python main.py compare  --steps 5 --plot
    python main.py intervention --steps 5 --plot

Graph parameters (all modes):
    --radius FLOAT         RGG connection radius (default: 0.30)
    --long-frac FLOAT      Fraction of nodes with long-range edges (default: 0.30)
    --long-k INT           Long-range edges per selected node (default: 2)
    --seed INT             Random seed (default: 42)
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "outputs"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from src.config import DEFAULT_TOPIC, DEFAULT_STEPS, NODE_NAMES
from src.network import create_graph, graph_summary, run_degroot


def _save_plot(filename: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR / filename


def _build_graph(args) -> nx.Graph:
    """Construct the RGG from CLI args and NODE_NAMES."""
    G = create_graph(
        names=NODE_NAMES,
        radius=args.radius,
        long_range_fraction=args.long_frac,
        long_range_k=args.long_k,
        seed=args.seed,
    )
    return G


def _print_summary(G: nx.Graph) -> None:
    s = graph_summary(G)
    print(f"\nRGG Summary:")
    print(f"  Nodes          : {s['nodes']}")
    print(f"  Edges total    : {s['edges']}  (local={s['local_edges']}, long-range={s['long_range_edges']})")
    print(f"  Avg degree     : {s['avg_degree']:.2f}")
    print(f"  Density        : {s['density']:.4f}")
    print(f"  Connected      : {s['is_connected']} ({s['components']} component(s))")
    print(f"  Long-range nodes: {s['long_range_nodes']}")
    print(f"  Block sizes    : {s['block_sizes']}")


BLOCK_COLORS = {0: "#3B82F6", 1: "#22C55E", 2: "#F97316", 3: "#EF4444"}
BLOCK_LABELS = {0: "left", 1: "center_left", 2: "center_right", 3: "right"}


def _plot_graph(G: nx.Graph, title: str, save_path: Path) -> None:
    """Draw the RGG with nodes colored by block and long-range edges dashed."""
    pos = {i: G.nodes[i]["pos"] for i in G.nodes()}
    blocks = nx.get_node_attributes(G, "block")
    colors = [BLOCK_COLORS[blocks[i]] for i in G.nodes()]

    local_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "local"]
    long_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "long_range"]

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_edges(G, pos, edgelist=local_edges, alpha=0.35, edge_color="#999", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=long_edges, alpha=0.6,
                           edge_color="#A855F7", style="dashed", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=180, ax=ax)

    # Short name labels (strip block prefix for readability)
    labels = {}
    for i in G.nodes():
        name = G.nodes[i]["name"]
        parts = name.split("_")
        # Strip leading block prefix(es): left_, center_left_, etc.
        if parts[0] in ("left", "right"):
            labels[i] = "_".join(parts[1:])
        elif len(parts) >= 2 and f"{parts[0]}_{parts[1]}" in ("center_left", "center_right"):
            labels[i] = "_".join(parts[2:])
        else:
            labels[i] = name
    nx.draw_networkx_labels(G, pos, labels, font_size=5, ax=ax)

    legend_handles = [
        mpatches.Patch(color=BLOCK_COLORS[b], label=BLOCK_LABELS[b]) for b in range(4)
    ]
    legend_handles += [
        plt.Line2D([0], [0], color="#999", label="local edge"),
        plt.Line2D([0], [0], color="#A855F7", linestyle="--", label="long-range edge"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Ideological position (left → right)")
    ax.set_ylabel("Engagement level")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    plt.close(fig)


# ─── Mode handlers ────────────────────────────────────────────────────────────

def main_graph(args):
    """Visualize the RGG topology (no LLM calls)."""
    print(f"Building RGG: {len(NODE_NAMES)} nodes, radius={args.radius}, "
          f"long_frac={args.long_frac}, long_k={args.long_k}")
    G = _build_graph(args)
    _print_summary(G)

    out = _save_plot("rgg_topology.png")
    _plot_graph(G,
                title=f"Random Geometric Graph  (r={args.radius}, long_frac={args.long_frac})",
                save_path=out)


def main_semantic(args):
    """Run semantic simulation only. ~5–15 min for 5 steps with 34 nodes."""
    from src.simulation import create_agents, run_semantic

    print(f"Building RGG ({len(NODE_NAMES)} nodes)...")
    G = _build_graph(args)
    _print_summary(G)

    print("Creating agents from nodes.json personas...")
    agents = create_agents(G, topic=args.topic, seed=args.seed)
    print("Running semantic simulation...")
    variances = run_semantic(G, agents, args.topic, steps=args.steps)

    print("\nSemantic variance over time:")
    for t, v in enumerate(variances):
        print(f"  t={t}: {v:.4f}")

    if args.plot:
        out = _save_plot("semantic_variance.png")
        plt.figure()
        plt.plot(variances, marker="o", color="steelblue")
        plt.xlabel("Timestep")
        plt.ylabel("Semantic Variance")
        plt.title(f"Semantic Opinion Dynamics (RGG + Long-Range)\nTopic: {args.topic}")
        plt.grid(True)
        plt.savefig(out, dpi=150)
        print(f"Saved {out}")
    return variances


def main_degroot(args):
    """Run DeGroot consensus only. Fast (seconds)."""
    print(f"Building RGG ({len(NODE_NAMES)} nodes)...")
    G = _build_graph(args)
    _print_summary(G)

    # Initial opinions: block position (0=left → 1=right) scaled to [0,1]
    blocks = nx.get_node_attributes(G, "block")
    n = G.number_of_nodes()
    rng = np.random.default_rng(args.seed)
    initial = np.array([(blocks[i] + 0.5) / 4.0 + rng.normal(0, 0.05) for i in range(n)])
    initial = np.clip(initial, 0.0, 1.0)

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
        plt.title("DeGroot Consensus (RGG + Long-Range)")
        plt.grid(True)
        plt.savefig(out, dpi=150)
        print(f"Saved {out}")
    return variances


def main_compare(args):
    """Run DeGroot (fast) then semantic (slow). Plot both for comparison."""
    from src.simulation import create_agents, run_semantic

    print(f"Building RGG ({len(NODE_NAMES)} nodes)...")
    G = _build_graph(args)
    _print_summary(G)

    blocks = nx.get_node_attributes(G, "block")
    n = G.number_of_nodes()
    initial_scalar = np.array([(blocks[i] + 0.5) / 4.0 for i in range(n)])

    print("Running DeGroot... (fast)")
    degroot_history = run_degroot(G, initial_scalar, steps=args.steps)
    degroot_var = [float(np.var(h)) for h in degroot_history]
    print("\nDeGroot variance:")
    for t, v in enumerate(degroot_var):
        print(f"  t={t}: {v:.4f}")

    print("\nRunning semantic simulation (this takes several minutes)...")
    agents = create_agents(G, topic=args.topic, seed=args.seed)
    semantic_var = run_semantic(G, agents, args.topic, steps=args.steps)
    print("\nSemantic variance:")
    for t, v in enumerate(semantic_var):
        print(f"  t={t}: {v:.4f}")

    if args.plot:
        out = _save_plot("semantic_vs_degroot.png")
        fig, ax = plt.subplots()
        ax.plot(degroot_var, marker="s", color="orange", label="DeGroot (scalar)")
        ax.plot(semantic_var, marker="o", color="steelblue", label="Semantic (LLM)")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Variance")
        ax.set_title(f"Semantic vs DeGroot — RGG + Long-Range\nTopic: {args.topic}")
        ax.legend()
        ax.grid(True)
        plt.savefig(out, dpi=150)
        print(f"Saved {out}")
    return semantic_var, degroot_var


def main_intervention(args):
    """Run semantic simulation with a disinformation bot."""
    from src.intervention import run_with_bot

    variances = run_with_bot(
        topic=args.topic,
        steps=args.steps,
        bot_post_prob=args.bot_prob,
        seed=args.seed,
        radius=args.radius,
        long_range_fraction=args.long_frac,
        long_range_k=args.long_k,
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
        plt.title("RGG: Resilience to Disinformation Bot")
        plt.grid(True)
        plt.savefig(out, dpi=150)
        print(f"Saved {out}")
    return variances


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _add_graph_args(p):
    """Add shared RGG topology arguments to a subparser."""
    p.add_argument("--radius", type=float, default=0.30,
                   help="RGG connection radius (default: 0.30)")
    p.add_argument("--long-frac", type=float, default=0.30,
                   help="Fraction of nodes with long-range edges (default: 0.30)")
    p.add_argument("--long-k", type=int, default=2,
                   help="Long-range edges per selected node (default: 2)")
    p.add_argument("--seed", type=int, default=42)


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Opinion Dynamics: LLM Agents on an RGG with Long-Range Connections"
    )
    sub = parser.add_subparsers(dest="mode", help="Mode to run")

    # graph: topology viz only
    p_g = sub.add_parser("graph", help="Visualize RGG topology (no LLM)")
    _add_graph_args(p_g)
    p_g.set_defaults(func=main_graph)

    # semantic
    p_sem = sub.add_parser("semantic", help="Run semantic (LLM) simulation")
    p_sem.add_argument("--topic", default=DEFAULT_TOPIC)
    p_sem.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p_sem.add_argument("--plot", action="store_true")
    _add_graph_args(p_sem)
    p_sem.set_defaults(func=main_semantic)

    # degroot
    p_deg = sub.add_parser("degroot", help="Run DeGroot consensus only")
    p_deg.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p_deg.add_argument("--plot", action="store_true")
    _add_graph_args(p_deg)
    p_deg.set_defaults(func=main_degroot)

    # compare
    p_cmp = sub.add_parser("compare", help="Run semantic and DeGroot, compare")
    p_cmp.add_argument("--topic", default=DEFAULT_TOPIC)
    p_cmp.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p_cmp.add_argument("--plot", action="store_true")
    _add_graph_args(p_cmp)
    p_cmp.set_defaults(func=main_compare)

    # intervention
    p_int = sub.add_parser("intervention", help="Disinformation bot intervention")
    p_int.add_argument("--topic", default=DEFAULT_TOPIC)
    p_int.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p_int.add_argument("--bot-prob", type=float, default=0.8)
    p_int.add_argument("--plot", action="store_true")
    _add_graph_args(p_int)
    p_int.set_defaults(func=main_intervention)

    args = parser.parse_args()
    if args.mode is None:
        parser.print_help()
        return 1
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())