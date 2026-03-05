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
from matplotlib.ticker import MaxNLocator

from src.config import (
    DEFAULT_EGT_BETA,
    DEFAULT_EGT_SWITCH_COST,
    DEFAULT_TOPIC,
    DEFAULT_STEPS,
    EGT_STRATEGIES,
    EGT_STRATEGY_PROTOTYPES,
)
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


def main_egt(args):
    """Run EGT dynamics using nodes.json personas and Fermi update."""
    from src.egt import run_egt
    from src.network import create_erdos_renyi_graph
    from src.nodes_data import load_nodes_data

    nodes_path = PROJECT_ROOT / "src" / "nodes.json"
    nodes_data = load_nodes_data(nodes_path)
    n = len(nodes_data)
    print(f"Loaded node personas: n={n} from {nodes_path}")
    print("Creating network (ER)...")
    G = create_erdos_renyi_graph(n=n, p=args.er_p, seed=args.seed)
    print(f"ER graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
    result = run_egt(
        G,
        steps=args.steps,
        beta=args.beta,
        switch_cost=args.switch_cost,
        seed=args.seed,
        nodes_data=nodes_data,
        topic=args.topic,
    )

    camp_names = ("left", "center_left", "center_right", "right")
    print("\nNode personas (node_id -> block | openness, stubbornness):")
    for p in result["personas"]:
        print(
            f"  {p['node_id']:>2}: {p['block']:<12} | openness={p['openness']:.2f}, "
            f"stubbornness={p['stubbornness']:.2f}"
        )

    print("\nEGT camp share over time:")
    for t, shares in enumerate(result["share_history"]):
        share_text = ", ".join(f"{camp_names[i]}={shares[i]:.2f}" for i in range(4))
        print(f"  t={t}: {share_text}")

    print("\nEGT polarization proxy over time (variance of strategy index):")
    for t, v in enumerate(result["polarization_history"]):
        print(f"  t={t}: {v:.4f}")

    if args.plot:
        out = _save_plot("egt_camp_shares.png")
        shares = np.array(result["share_history"])
        plt.figure()
        for i, camp in enumerate(camp_names):
            plt.plot(shares[:, i], marker="o", label=camp)
        plt.xlabel("Timestep")
        plt.ylabel("Population Share")
        plt.title(f"EGT Camp Share Dynamics (nodes.json, ER)\nTopic: {args.topic}")
        plt.ylim(0.0, 1.0)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.grid(True)
        plt.legend()
        plt.savefig(out, dpi=150)
        print(f"\nSaved {out}")
    return result


def main_hybrid(args):
    """Run fully black-box LLM updates; evaluate only at end with SBERT."""
    from src.hybrid_egt import run_hybrid_llm_egt

    print("Creating network (SBM: 20 nodes, 4 blocks)...")
    G = create_graph(seed=args.seed)
    print("Running black-box LLM simulation...")
    result = run_hybrid_llm_egt(
        G=G,
        topic=args.topic,
        steps=args.steps,
        seed=args.seed,
    )

    print("\nFinal performance (SBERT-based):")
    print(f"  Initial semantic variance: {result['initial_semantic_variance']:.4f}")
    print(f"  Final semantic variance:   {result['final_semantic_variance']:.4f}")
    print(f"  Semantic drift:            {result['semantic_drift']:.4f}")

    if args.plot:
        var_out = _save_plot("hybrid_final_performance.png")
        plt.figure()
        labels = ["initial_variance", "final_variance", "semantic_drift"]
        values = [
            result["initial_semantic_variance"],
            result["final_semantic_variance"],
            result["semantic_drift"],
        ]
        plt.bar(labels, values, color=["steelblue", "purple", "darkorange"])
        plt.ylabel("Value")
        plt.title(f"Black-box LLM Final Performance\nTopic: {args.topic}")
        plt.grid(True)
        plt.savefig(var_out, dpi=150)
        print(f"Saved {var_out}")
    return result


def _step_directions(values: list[float], eps: float = 1e-6) -> list[str]:
    """
    Convert variance deltas into per-step direction labels.
    """
    labels = []
    for i in range(1, len(values)):
        delta = values[i] - values[i - 1]
        if delta > eps:
            labels.append("diverge")
        elif delta < -eps:
            labels.append("converge")
        else:
            labels.append("stable")
    return labels


def _first_stable_step(values: list[float], eps: float = 1e-3) -> int | None:
    """First timestep where |delta| <= eps."""
    for t in range(1, len(values)):
        if abs(values[t] - values[t - 1]) <= eps:
            return t
    return None


def _infer_group_from_name(name: str) -> str:
    """
    Infer political group label from node name prefix.
    """
    if name.startswith("center_left_"):
        return "center_left"
    if name.startswith("center_right_"):
        return "center_right"
    if name.startswith("left_"):
        return "left"
    if name.startswith("right_"):
        return "right"
    return "unknown"


def _group_semantic_change_series(
    nodes_data: list[dict],
    opinions_history: list[list[str]],
) -> dict[str, list[float]]:
    """
    Compute per-step mean semantic drift (vs t=0) for each political group.
    """
    from src.measurement import embed_opinions

    groups = ("left", "center_left", "center_right", "right")
    grouped_initial = {g: [] for g in groups}
    grouped_indices = {g: [] for g in groups}

    for i, rec in enumerate(nodes_data):
        group = _infer_group_from_name(str(rec.get("name", "")))
        if group not in grouped_indices:
            continue
        grouped_initial[group].append(str(rec.get("initial", "")))
        grouped_indices[group].append(i)

    initial_emb_by_group = {}
    for g in groups:
        if not grouped_initial[g]:
            initial_emb_by_group[g] = None
        else:
            initial_emb_by_group[g] = embed_opinions(grouped_initial[g])

    result = {g: [] for g in groups}
    for opinions_t in opinions_history:
        for g in groups:
            idxs = grouped_indices[g]
            if not idxs or initial_emb_by_group[g] is None:
                result[g].append(0.0)
                continue
            group_current = [opinions_t[i] for i in idxs]
            current_emb = embed_opinions(group_current)
            drift = float(np.mean(np.linalg.norm(current_emb - initial_emb_by_group[g], axis=1)))
            result[g].append(drift)
    return result


def _camp_share_series_prefix_then_semantic(
    nodes_data: list[dict],
    opinions_history: list[list[str]],
) -> dict[str, list[float]]:
    """
    Camp shares over time:
      - t=0: inferred from node name prefix (fixed initialization)
      - t>=1: semantically classified from each step's opinions via SBERT
    """
    from src.measurement import embed_opinions

    camps = list(EGT_STRATEGIES)
    n_nodes = len(nodes_data)
    n_steps = len(opinions_history)
    result = {c: [0.0] * n_steps for c in camps}

    # Keep t=0 aligned with explicit initialization from node name prefixes.
    t0_counts = {c: 0 for c in camps}
    for rec in nodes_data:
        group = _infer_group_from_name(str(rec.get("name", "")))
        if group in t0_counts:
            t0_counts[group] += 1
    for c in camps:
        result[c][0] = (t0_counts[c] / n_nodes) if n_nodes > 0 else 0.0

    # t>=1: classify each node's current opinion to nearest camp prototype.
    prototype_texts = [EGT_STRATEGY_PROTOTYPES[c] for c in camps]
    prototype_emb = embed_opinions(prototype_texts)
    for t in range(1, n_steps):
        opinions_t = opinions_history[t]
        if not opinions_t:
            continue
        emb_t = embed_opinions(opinions_t)
        dists = np.linalg.norm(emb_t[:, None, :] - prototype_emb[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        counts_t = np.bincount(labels, minlength=len(camps))
        denom = len(opinions_t)
        for idx, c in enumerate(camps):
            result[c][t] = (float(counts_t[idx]) / denom) if denom > 0 else 0.0
    return result


def main_topology(args):
    """
    Run semantic simulation on ER and Random Geometric graphs using nodes.json personas.
    """
    from src.network import create_erdos_renyi_graph, create_random_geometric_graph
    from src.nodes_data import load_nodes_data
    from src.simulation import create_agents_from_nodes_data, run_semantic

    nodes_path = PROJECT_ROOT / "src" / "nodes.json"
    nodes_data = load_nodes_data(nodes_path)
    n = len(nodes_data)
    print(f"Loaded node personas: n={n} from {nodes_path}")

    er_var = None
    rgg_var = None
    er_dir = []
    rgg_dir = []
    er_group_change_series = {g: [] for g in ("left", "center_left", "center_right", "right")}
    rgg_group_change_series = {g: [] for g in ("left", "center_left", "center_right", "right")}
    er_group_change = {g: 0.0 for g in ("left", "center_left", "center_right", "right")}
    rgg_group_change = {g: 0.0 for g in ("left", "center_left", "center_right", "right")}
    er_camp_share_series = None
    rgg_camp_share_series = None

    if args.network in ("er", "both"):
        print("\nBuilding Erdos-Renyi graph...")
        G_er = create_erdos_renyi_graph(n=n, p=args.er_p, seed=args.seed)
        print(f"ER graph: nodes={G_er.number_of_nodes()}, edges={G_er.number_of_edges()}")
        agents_er = create_agents_from_nodes_data(G_er, nodes_data, topic=args.topic)
        er_opinion_history = [[a.current_opinion for a in agents_er]]

        def _capture_er(_t, agents_state):
            er_opinion_history.append([a.current_opinion for a in agents_state])

        er_var = run_semantic(G_er, agents_er, args.topic, steps=args.steps, on_step=_capture_er)
        er_dir = _step_directions(er_var)
        er_group_change_series = _group_semantic_change_series(nodes_data, er_opinion_history)
        er_group_change = {g: er_group_change_series[g][-1] for g in er_group_change_series}
        er_camp_share_series = _camp_share_series_prefix_then_semantic(nodes_data, er_opinion_history)

    if args.network in ("rgg", "both"):
        print("\nBuilding Random Geometric graph...")
        G_rgg = create_random_geometric_graph(n=n, radius=args.rgg_radius, seed=args.seed)
        print(f"RGG graph: nodes={G_rgg.number_of_nodes()}, edges={G_rgg.number_of_edges()}")
        agents_rgg = create_agents_from_nodes_data(G_rgg, nodes_data, topic=args.topic)
        rgg_opinion_history = [[a.current_opinion for a in agents_rgg]]

        def _capture_rgg(_t, agents_state):
            rgg_opinion_history.append([a.current_opinion for a in agents_state])

        rgg_var = run_semantic(G_rgg, agents_rgg, args.topic, steps=args.steps, on_step=_capture_rgg)
        rgg_dir = _step_directions(rgg_var)
        rgg_group_change_series = _group_semantic_change_series(nodes_data, rgg_opinion_history)
        rgg_group_change = {g: rgg_group_change_series[g][-1] for g in rgg_group_change_series}
        rgg_camp_share_series = _camp_share_series_prefix_then_semantic(nodes_data, rgg_opinion_history)

    if er_var is not None:
        print("\nPer-step direction (ER):")
        for step, label in enumerate(er_dir, start=1):
            print(f"  t={step}: {label} (variance={er_var[step]:.4f})")
        er_stable = _first_stable_step(er_var)
        print(f"\nER first stable step (|delta variance|<=1e-3): {er_stable}")
    if rgg_var is not None:
        print("\nPer-step direction (RGG):")
        for step, label in enumerate(rgg_dir, start=1):
            print(f"  t={step}: {label} (variance={rgg_var[step]:.4f})")
        rgg_stable = _first_stable_step(rgg_var)
        print(f"\nRGG first stable step (|delta variance|<=1e-3): {rgg_stable}")

    if er_var is not None:
        print("\nFinal semantic change by group (ER):")
        for g in ("left", "center_left", "center_right", "right"):
            print(f"  {g}: {er_group_change[g]:.4f}")
    if rgg_var is not None:
        print("\nFinal semantic change by group (RGG):")
        for g in ("left", "center_left", "center_right", "right"):
            print(f"  {g}: {rgg_group_change[g]:.4f}")

    if args.plot:
        groups = ("left", "center_left", "center_right", "right")
        camps = list(EGT_STRATEGIES)

        if er_var is not None:
            t_axis_er = np.arange(0, len(er_var))
            out_er_group = _save_plot("topology_er_group_final_change.png")
            out_er_share = _save_plot("topology_er_camp_shares.png")

            plt.figure()
            plt.plot(er_var, marker="o", label="ER variance")
            plt.xlabel("Timestep")
            plt.ylabel("Semantic Variance")
            plt.title(f"ER Semantic Dynamics\nTopic: {args.topic}")
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.grid(True)
            plt.legend()
            plt.savefig(_save_plot("topology_variance_comparison.png"), dpi=150)

            er_delta = np.diff(np.array(er_var))
            x = np.arange(1, len(er_var))
            plt.figure()
            plt.bar(x, er_delta, width=0.5, label="ER delta variance")
            plt.axhline(0.0, color="black", linewidth=1)
            plt.xlabel("Timestep")
            plt.ylabel("Delta Variance (t - (t-1))")
            plt.title("ER: Converge (<0) vs Diverge (>0) by Step")
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.grid(True)
            plt.legend()
            plt.savefig(_save_plot("topology_converge_diverge_steps.png"), dpi=150)

            plt.figure()
            for g in groups:
                plt.plot(t_axis_er, er_group_change_series[g], marker="o", label=g)
            plt.xlabel("Timestep")
            plt.ylabel("Mean Semantic Drift vs t=0")
            plt.title("ER: Per-step Change by Group")
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.grid(True)
            plt.legend()
            plt.savefig(out_er_group, dpi=150)
            print(f"Saved {out_er_group}")

            plt.figure()
            for c in camps:
                plt.plot(t_axis_er, er_camp_share_series[c], marker="o", label=c)
            plt.xlabel("Timestep")
            plt.ylabel("Population Share")
            plt.title("ER Camp Share Dynamics (t0 prefix, t>=1 semantic)")
            plt.ylim(0.0, 1.0)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.grid(True)
            plt.legend()
            plt.savefig(out_er_share, dpi=150)
            print(f"Saved {out_er_share}")

        if rgg_var is not None:
            t_axis_rgg = np.arange(0, len(rgg_var))
            out_rgg_group = _save_plot("topology_rgg_group_final_change.png")
            out_rgg_share = _save_plot("topology_rgg_camp_shares.png")

            if er_var is None:
                plt.figure()
                plt.plot(rgg_var, marker="s", label="RGG variance")
                plt.xlabel("Timestep")
                plt.ylabel("Semantic Variance")
                plt.title(f"RGG Semantic Dynamics\nTopic: {args.topic}")
                plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
                plt.grid(True)
                plt.legend()
                plt.savefig(_save_plot("topology_variance_comparison.png"), dpi=150)

                rgg_delta = np.diff(np.array(rgg_var))
                x = np.arange(1, len(rgg_var))
                plt.figure()
                plt.bar(x, rgg_delta, width=0.5, label="RGG delta variance")
                plt.axhline(0.0, color="black", linewidth=1)
                plt.xlabel("Timestep")
                plt.ylabel("Delta Variance (t - (t-1))")
                plt.title("RGG: Converge (<0) vs Diverge (>0) by Step")
                plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
                plt.grid(True)
                plt.legend()
                plt.savefig(_save_plot("topology_converge_diverge_steps.png"), dpi=150)

            plt.figure()
            for g in groups:
                plt.plot(t_axis_rgg, rgg_group_change_series[g], marker="o", label=g)
            plt.xlabel("Timestep")
            plt.ylabel("Mean Semantic Drift vs t=0")
            plt.title("RGG: Per-step Change by Group")
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.grid(True)
            plt.legend()
            plt.savefig(out_rgg_group, dpi=150)
            print(f"Saved {out_rgg_group}")

            plt.figure()
            for c in camps:
                plt.plot(t_axis_rgg, rgg_camp_share_series[c], marker="o", label=c)
            plt.xlabel("Timestep")
            plt.ylabel("Population Share")
            plt.title("RGG Camp Share Dynamics (t0 prefix, t>=1 semantic)")
            plt.ylim(0.0, 1.0)
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.grid(True)
            plt.legend()
            plt.savefig(out_rgg_share, dpi=150)
            print(f"Saved {out_rgg_share}")

    return {
        "er_variance": er_var,
        "rgg_variance": rgg_var,
        "er_direction": er_dir,
        "rgg_direction": rgg_dir,
        "er_group_change_series": er_group_change_series,
        "rgg_group_change_series": rgg_group_change_series,
        "er_camp_share_series": er_camp_share_series,
        "rgg_camp_share_series": rgg_camp_share_series,
        "er_group_change": er_group_change,
        "rgg_group_change": rgg_group_change,
    }


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

    p_egt = sub.add_parser("egt", help="Run evolutionary game simulation with nodes.json personas")
    p_egt.add_argument("--topic", default="government environment regulations")
    p_egt.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p_egt.add_argument("--seed", type=int, default=42)
    p_egt.add_argument("--er-p", type=float, default=0.2, help="Erdos-Renyi edge probability")
    p_egt.add_argument("--beta", type=float, default=DEFAULT_EGT_BETA)
    p_egt.add_argument("--switch-cost", type=float, default=DEFAULT_EGT_SWITCH_COST)
    p_egt.add_argument("--plot", action="store_true")
    p_egt.set_defaults(func=main_egt)

    p_hybrid = sub.add_parser("hybrid", help="Run fully black-box LLM simulation")
    p_hybrid.add_argument("--topic", default=DEFAULT_TOPIC)
    p_hybrid.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p_hybrid.add_argument("--seed", type=int, default=42)
    p_hybrid.add_argument("--plot", action="store_true")
    p_hybrid.set_defaults(func=main_hybrid)

    p_top = sub.add_parser("topology", help="Run ER vs RGG using src/nodes.json personas")
    p_top.add_argument("--topic", default=DEFAULT_TOPIC)
    p_top.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p_top.add_argument("--seed", type=int, default=42)
    p_top.add_argument("--er-p", type=float, default=0.20, help="Edge probability for ER graph")
    p_top.add_argument(
        "--rgg-radius",
        type=float,
        default=0.35,
        help="Connection radius for random geometric graph",
    )
    p_top.add_argument("--plot", action="store_true")
    p_top.add_argument(
        "--network",
        choices=("er", "rgg", "both"),
        default="both",
        help="Which topology to run",
    )
    p_top.set_defaults(func=main_topology)

    args = parser.parse_args()
    if args.mode is None:
        parser.print_help()
        return 1
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
