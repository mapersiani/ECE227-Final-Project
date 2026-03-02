#!/usr/bin/env python3
"""
Semantic Opinion Dynamics: LLM Agents on a Random Geometric Graph with Long-Range Connections

Agents' OPINIONS update every step; PERSONAS drift every N steps from social influence.
Full graph metrics (clustering, modularity, betweenness, drift) are collected each step.

Usage:
    python main.py graph                                  # Visualize topology (no LLM)
    python main.py semantic --steps 5 --plot              # Full run + analysis
    python main.py degroot  --steps 5 --plot              # Fast scalar baseline
    python main.py compare  --steps 5 --plot              # Both side-by-side
    python main.py intervention --steps 5 --plot          # Disinformation bot

Key flags (all modes):
    --radius FLOAT          RGG radius (default 0.30)
    --long-frac FLOAT       Fraction of long-range nodes (default 0.30)
    --long-k INT            Long-range edges per node (default 2)
    --persona-drift-every N Persona update frequency in steps (0=off, default 2)
    --no-metrics            Skip per-step graph metrics (faster, less data)
    --seed INT              Random seed (default 42)
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "outputs"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from src.config import DEFAULT_TOPIC, DEFAULT_STEPS, NODE_NAMES
from src.network import create_graph, graph_summary, run_degroot


def _save(filename: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR / filename


def _build_graph(args) -> nx.Graph:
    return create_graph(
        names=NODE_NAMES,
        radius=args.radius,
        long_range_fraction=args.long_frac,
        long_range_k=args.long_k,
        seed=args.seed,
    )


def _print_summary(G: nx.Graph) -> None:
    s = graph_summary(G)
    print(f"\nRGG — {s['nodes']} nodes  {s['edges']} edges "
          f"(local={s['local_edges']}, long-range={s['long_range_edges']})")
    print(f"  avg_degree={s['avg_degree']:.2f}  density={s['density']:.4f}  "
          f"connected={s['is_connected']}  long_range_nodes={s['long_range_nodes']}")
    print(f"  block_sizes={s['block_sizes']}")


BLOCK_COLORS = {0: "#3B82F6", 1: "#22C55E", 2: "#F97316", 3: "#EF4444"}
BLOCK_LABELS = {0: "left", 1: "center_left", 2: "center_right", 3: "right"}


def _plot_topology(G, title, save_path):
    pos = {i: G.nodes[i]["pos"] for i in G.nodes()}
    blocks = nx.get_node_attributes(G, "block")
    colors = [BLOCK_COLORS[blocks[i]] for i in G.nodes()]
    local_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "local"]
    long_edges  = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "long_range"]

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_edges(G, pos, edgelist=local_edges, alpha=0.3, edge_color="#aaa", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=long_edges, alpha=0.6, edge_color="#A855F7",
                           style="dashed", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=180, ax=ax)

    labels = {}
    for i in G.nodes():
        parts = G.nodes[i].get("name", str(i)).split("_")
        if parts[0] in ("left", "right"):
            labels[i] = "_".join(parts[1:])
        elif len(parts) >= 2 and f"{parts[0]}_{parts[1]}" in ("center_left", "center_right"):
            labels[i] = "_".join(parts[2:])
        else:
            labels[i] = G.nodes[i].get("name", str(i))
    nx.draw_networkx_labels(G, pos, labels, font_size=5, ax=ax)

    legend = [mpatches.Patch(color=BLOCK_COLORS[b], label=BLOCK_LABELS[b]) for b in range(4)]
    legend += [
        plt.Line2D([0], [0], color="#aaa", label="local edge"),
        plt.Line2D([0], [0], color="#A855F7", linestyle="--", label="long-range edge"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8)
    ax.set_title(title); ax.set_xlabel("Ideological position"); ax.set_ylabel("Engagement level")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    plt.close(fig)


# ─── Mode handlers ─────────────────────────────────────────────────────────────

def main_graph(args):
    G = _build_graph(args)
    _print_summary(G)
    _plot_topology(G,
                   title=f"RGG Topology  (r={args.radius}, long_frac={args.long_frac})",
                   save_path=_save("rgg_topology.png"))


def main_semantic(args):
    from src.simulation import create_agents, run_semantic
    from src.analysis import run_full_analysis

    print(f"Building RGG ({len(NODE_NAMES)} nodes)...")
    G = _build_graph(args)
    _print_summary(G)

    print("Creating agents from nodes.json personas...")
    agents = create_agents(G, topic=args.topic, seed=args.seed)

    print(f"Running semantic simulation  "
          f"(steps={args.steps}, persona_drift_every={args.persona_drift_every}, "
          f"metrics={'yes' if not args.no_metrics else 'no'})...")

    record = run_semantic(
        G, agents, args.topic,
        steps=args.steps,
        persona_drift_every=args.persona_drift_every,
        compute_metrics=not args.no_metrics,
        show_progress=True,
    )

    if args.plot:
        run_full_analysis(record, agents, G, output_dir=OUTPUT_DIR)
    else:
        print("\nFinal summary:")
        if record.step_snapshots:
            last = record.step_snapshots[-1]
            for k, v in last.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")

    return record


def main_degroot(args):
    print(f"Building RGG ({len(NODE_NAMES)} nodes)...")
    G = _build_graph(args)
    _print_summary(G)

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
        out = _save("degroot_variance.png")
        plt.figure()
        plt.plot(variances, marker="s", color="orange", linewidth=2)
        plt.xlabel("Timestep"); plt.ylabel("Opinion Variance")
        plt.title("DeGroot Consensus — RGG + Long-Range")
        plt.grid(True); plt.savefig(out, dpi=150); print(f"Saved {out}")
        plt.close()
    return variances


def main_compare(args):
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

    print("Running semantic simulation...")
    agents = create_agents(G, topic=args.topic, seed=args.seed)
    record = run_semantic(
        G, agents, args.topic,
        steps=args.steps,
        persona_drift_every=args.persona_drift_every,
        compute_metrics=not args.no_metrics,
        show_progress=True,
    )
    semantic_var = record.scalar_series("opinion_variance")

    print("\nDeGroot variance:", [f"{v:.4f}" for v in degroot_var])
    print("Semantic variance:", [f"{v:.4f}" for v in semantic_var])

    if args.plot:
        out = _save("semantic_vs_degroot.png")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(degroot_var, marker="s", color="orange", label="DeGroot (scalar)", linewidth=2)
        ax.plot(semantic_var, marker="o", color="steelblue", label="Semantic + Persona Drift (LLM)", linewidth=2)
        ax.set_xlabel("Timestep"); ax.set_ylabel("Variance")
        ax.set_title(f"Semantic vs DeGroot — RGG\nTopic: {args.topic}")
        ax.legend(); ax.grid(True)
        plt.savefig(out, dpi=150); print(f"Saved {out}")
        plt.close(fig)

        if not args.no_metrics:
            from src.analysis import run_full_analysis
            run_full_analysis(record, agents, G, output_dir=OUTPUT_DIR)

    return record, degroot_var


def main_intervention(args):
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
        out = _save("intervention_comparison.png")
        plt.figure()
        plt.plot(variances, marker="o", linewidth=2)
        plt.xlabel("Timestep"); plt.ylabel("Semantic Variance")
        plt.title("RGG: Resilience to Disinformation Bot")
        plt.grid(True); plt.savefig(out, dpi=150); print(f"Saved {out}")
        plt.close()
    return variances


# ─── CLI ───────────────────────────────────────────────────────────────────────

def _add_graph_args(p):
    p.add_argument("--radius",    type=float, default=0.30)
    p.add_argument("--long-frac", type=float, default=0.30)
    p.add_argument("--long-k",    type=int,   default=2)
    p.add_argument("--seed",      type=int,   default=42)

def _add_sim_args(p):
    _add_graph_args(p)
    p.add_argument("--persona-drift-every", type=int, default=2,
                   help="Run persona drift every N steps (0=off)")
    p.add_argument("--no-metrics", action="store_true",
                   help="Skip per-step graph metrics (faster)")


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Opinion Dynamics with Persona Drift — RGG Network"
    )
    sub = parser.add_subparsers(dest="mode")

    p = sub.add_parser("graph", help="Visualize RGG topology (no LLM)")
    _add_graph_args(p); p.set_defaults(func=main_graph)

    p = sub.add_parser("semantic", help="Full semantic sim with persona drift + analysis")
    p.add_argument("--topic",  default=DEFAULT_TOPIC)
    p.add_argument("--steps",  type=int, default=DEFAULT_STEPS)
    p.add_argument("--plot",   action="store_true")
    _add_sim_args(p); p.set_defaults(func=main_semantic)

    p = sub.add_parser("degroot", help="DeGroot scalar consensus (fast baseline)")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p.add_argument("--plot",  action="store_true")
    _add_graph_args(p); p.set_defaults(func=main_degroot)

    p = sub.add_parser("compare", help="Semantic + DeGroot side-by-side")
    p.add_argument("--topic",  default=DEFAULT_TOPIC)
    p.add_argument("--steps",  type=int, default=DEFAULT_STEPS)
    p.add_argument("--plot",   action="store_true")
    _add_sim_args(p); p.set_defaults(func=main_compare)

    p = sub.add_parser("intervention", help="Disinformation bot resilience study")
    p.add_argument("--topic",    default=DEFAULT_TOPIC)
    p.add_argument("--steps",    type=int,   default=DEFAULT_STEPS)
    p.add_argument("--bot-prob", type=float, default=0.8)
    p.add_argument("--plot",     action="store_true")
    _add_graph_args(p); p.set_defaults(func=main_intervention)

    args = parser.parse_args()
    if args.mode is None:
        parser.print_help()
        return 1
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())