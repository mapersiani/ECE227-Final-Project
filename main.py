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
FIXED_TOPIC = "government environment regulations"

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.ticker import MaxNLocator

from src.config import (
    DEFAULT_TOPIC,
    DEFAULT_STEPS,
    DEFAULT_SEED,
    DEFAULT_ER_EDGE_PROB,
    DEFAULT_BOT_POST_PROB,
    DEFAULT_BOT_POST_MULTIPLIER,
    DEFAULT_BOT_TARGET_FRAC,
    DEFAULT_BOT_CONNECT_PROB,
    RGG_RADIUS,
    LONG_RANGE_FRACTION,
    SEED_LIST,
)
from src.network import create_erdos_renyi_graph

CAMP_LABELS = ("democracy", "republican")


def _save_plot(filename: str) -> Path:
    """Create outputs/ if needed and return path for saving plot."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR / filename


def _nodes_path_by_dataset(dataset: str) -> Path:
    if dataset == "senate":
        return PROJECT_ROOT / "src" / "senate_nodes.json"
    return PROJECT_ROOT / "src" / "nodes.json"


def _save_network_animation(
    G: nx.Graph,
    opinions_history: list[list[str]],
    topic: str,
    out_name: str,
    classify_fn=None,
    layout_seed: int = 42,
    fps: int = 2,
) -> Path:
    """
    Save a GIF animation of network evolution with side colors.
    """
    from src.measurement import classify_side_labels, embed_opinions

    out = _save_plot(out_name)
    color_map = {
        "democracy": "#1f77b4",
        "democrat": "#1f77b4",
        "republican": "#d62728",
    }
    node_order = sorted(list(G.nodes()))

    pos_attr = nx.get_node_attributes(G, "pos")
    if len(pos_attr) == G.number_of_nodes():
        pos = {n: pos_attr[n] for n in node_order}
    else:
        pos = nx.spring_layout(G, seed=layout_seed)
        pos = {n: pos[n] for n in node_order}

    n = len(node_order)
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="#f2f2f2")
    if len(pos_attr) != G.number_of_nodes():
        pos = nx.spring_layout(G, seed=layout_seed, k=1.8 / np.sqrt(max(1, n)), iterations=400)
        pos = {n: pos[n] for n in node_order}

    def _draw_frame(t: int):
        ax.clear()
        ax.set_facecolor("#f2f2f2")
        opinions_t = opinions_history[t]
        if classify_fn is None:
            labels_t = classify_side_labels(embed_opinions(opinions_t))
        else:
            labels_t = classify_fn(opinions_t)
        node_colors = [color_map.get(labels_t[i], "#7f7f7f") for i in range(len(node_order))]
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#b2182b", alpha=0.20, width=0.35)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=node_order,
            node_color=node_colors,
            node_size=18,
            linewidths=0.0,
            edgecolors="none",
            ax=ax,
        )
        ax.set_title(f"t={t}", fontsize=12, color="#333333")
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.02)
        ax.set_axis_off()

    ani = animation.FuncAnimation(
        fig,
        _draw_frame,
        frames=len(opinions_history),
        interval=max(200, int(1000 / max(1, fps))),
        repeat=False,
    )
    ani.save(out, writer=animation.PillowWriter(fps=max(1, fps)))
    plt.close(fig)
    return out


def _side_from_name_general(name: str) -> str:
    """Infer class label from the token before the first underscore."""
    token = str(name).split("_", 1)[0].strip().lower()
    if token == "republican":
        return "republican"
    return "democracy"


def _build_classifier_from_nodes(nodes_data: list[dict]):
    """
    Build an opinion->label classifier using nearest persona prototype
    from the provided nodes dataset.
    """
    from src.measurement import embed_opinions

    def _label_from_record(rec: dict) -> str:
        party = str(rec.get("party", "")).strip().lower()
        if party.startswith("rep"):
            return "republican"
        if party.startswith("dem"):
            return "democracy"
        return _side_from_name_general(str(rec.get("name", "")))

    labels = [_label_from_record(rec) for rec in nodes_data]
    proto_texts = [str(rec.get("initial") or rec.get("prompt") or "neutral policy statement") for rec in nodes_data]
    proto_emb = embed_opinions(proto_texts)

    def _classify(opinions_t: list[str]) -> list[str]:
        if not opinions_t:
            return []
        emb_t = embed_opinions(opinions_t)
        emb_norm = emb_t / (np.linalg.norm(emb_t, axis=1, keepdims=True) + 1e-8)
        proto_norm = proto_emb / (np.linalg.norm(proto_emb, axis=1, keepdims=True) + 1e-8)
        sims = emb_norm @ proto_norm.T
        nn = np.argmax(sims, axis=1)
        return [labels[int(i)] for i in nn]

    return _classify


def main_semantic(args):
    """Run semantic simulation only. Agents update opinions via LLM (Ollama). ~5–10 min for 5 steps."""
    from src.simulation import create_agents, run_semantic

    print("Creating network (ER: 20 nodes)...")
    G = create_erdos_renyi_graph(n=20, p=DEFAULT_ER_EDGE_PROB, seed=args.seed)
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
        plt.title(f"Semantic Opinion Dynamics (ER)\nTopic: {args.topic}")
        plt.grid(True)
        plt.savefig(out, dpi=150)
        print(f"\nSaved {out}")
    return variances


def main_degroot(args):
    """Run DeGroot consensus only. Fast (seconds). Scalar opinions average toward consensus."""
    from src.network import run_degroot

    print("Creating network (ER: 20 nodes)...")
    G = create_erdos_renyi_graph(n=20, p=DEFAULT_ER_EDGE_PROB, seed=args.seed)
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
        plt.title("DeGroot Consensus (ER)")
        plt.grid(True)
        plt.savefig(out, dpi=150)
        print(f"\nSaved {out}")
    return variances


def main_compare(args):
    """Run DeGroot (fast) then semantic (slow). Plot both on one figure for comparison."""
    from src.network import run_degroot
    from src.simulation import create_agents, run_semantic

    print("Creating network (ER: 20 nodes)...")
    G = create_erdos_renyi_graph(n=20, p=DEFAULT_ER_EDGE_PROB, seed=args.seed)
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
        ax.set_title(f"Semantic vs DeGroot Opinion Dynamics (ER)\nTopic: {args.topic}")
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
        bot_post_multiplier=args.bot_mult,
        bot_connect_prob=args.bot_connect_prob,
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
        plt.title("ER: Resilience to Disinformation Bot")
        plt.grid(True)
        plt.savefig(out, dpi=150)
        print(f"\nSaved {out}")
    return variances


def main_topology_intervention(args):
    """Evaluate ER resilience to semantic drift induced by a high-frequency bot."""
    from src.intervention import run_topology_resilience_with_bot
    from src.nodes_data import load_nodes_data

    nodes_path = PROJECT_ROOT / "src" / "nodes.json"
    nodes_data = load_nodes_data(nodes_path)
    print(f"Loaded node personas: n={len(nodes_data)} from {nodes_path}")
    print(
        "Running topology resilience study with bot "
        f"(post_prob={args.bot_prob}, post_multiplier={args.bot_mult})..."
    )

    result = run_topology_resilience_with_bot(
        nodes_data=nodes_data,
        topic=args.topic,
        steps=args.steps,
        seed=args.seed,
        er_p=args.er_p,
        bot_post_prob=args.bot_prob,
        bot_post_multiplier=args.bot_mult,
        bot_connect_prob=args.bot_connect_prob,
    )

    er = result["er"]
    print(f"\nER graph: nodes={er['n_nodes']}, edges={er['n_edges']}")
    print("Per-step bot-induced semantic drift (vs no-bot baseline):")
    for t, d in enumerate(er["drift_series"]):
        print(f"  t={t}: {d:.4f}")
    print(f"Final drift:      {er['final_drift']:.4f}")
    print(f"Cumulative drift: {er['cumulative_drift']:.4f}")

    if args.plot:
        out_drift = _save_plot("topology_bot_resilience_drift.png")
        out_er = _save_plot("topology_er_bot_vs_baseline_variance.png")

        plt.figure()
        plt.plot(er["drift_series"], marker="o", label="ER drift")
        plt.xlabel("Timestep")
        plt.ylabel("Mean Semantic Drift (bot vs baseline)")
        plt.title(f"ER Resilience to Disinformation Bot\nTopic: {args.topic}")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.grid(True)
        plt.legend()
        plt.savefig(out_drift, dpi=150)
        print(f"\nSaved {out_drift}")

        plt.figure()
        plt.plot(er["baseline_variance"], marker="o", label="ER baseline")
        plt.plot(er["bot_variance"], marker="x", label="ER + bot")
        plt.xlabel("Timestep")
        plt.ylabel("Semantic Variance")
        plt.title("ER: Baseline vs Disinformation Bot")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.grid(True)
        plt.legend()
        plt.savefig(out_er, dpi=150)
        print(f"Saved {out_er}")
    return result


def main_sw_sf_intervention(args):
    """Compare small-world vs scale-free resilience under disinformation bot across SEED_LIST."""
    from src.intervention import run_sw_sf_resilience_with_bot
    from src.nodes_data import load_nodes_data

    seeds = list(SEED_LIST)
    nodes_path = _nodes_path_by_dataset(args.dataset)
    nodes_data = load_nodes_data(nodes_path)
    print(f"Loaded node personas: n={len(nodes_data)} from {nodes_path}; seeds={seeds}")
    print(
        "Running small-world vs scale-free intervention "
        f"(bot_prob={args.bot_prob}, bot_mult={args.bot_mult})..."
    )
    runs: list[dict] = []
    for seed in seeds:
        run = run_sw_sf_resilience_with_bot(
            nodes_data=nodes_data,
            topic=args.topic,
            steps=args.steps,
            seed=seed,
            sw_radius=args.sw_radius,
            sw_long_prob=args.sw_long_prob,
            sf_m=args.sf_m,
            bot_post_prob=args.bot_prob,
            bot_post_multiplier=args.bot_mult,
            bot_connect_prob=args.bot_connect_prob,
        )
        run["seed"] = seed
        runs.append(run)
        print(
            f"  seed={seed}: SW final={run['small_world']['final_drift']:.4f}, "
            f"SF final={run['scale_free']['final_drift']:.4f}"
        )

    topologies = ("small_world", "scale_free")
    summary: dict[str, dict[str, float]] = {}
    for topo in topologies:
        final_arr = np.array([r[topo]["final_drift"] for r in runs], dtype=float)
        cum_arr = np.array([r[topo]["cumulative_drift"] for r in runs], dtype=float)
        summary[topo] = {
            "final_mean": float(np.mean(final_arr)),
            "final_std": float(np.std(final_arr)),
            "cum_mean": float(np.mean(cum_arr)),
            "cum_std": float(np.std(cum_arr)),
        }

    better = min(topologies, key=lambda k: summary[k]["final_mean"])
    print(f"\nMore resilient (by mean final drift): {better}")
    print(
        "  small_world final mean±std = "
        f"{summary['small_world']['final_mean']:.4f} ± {summary['small_world']['final_std']:.4f}"
    )
    print(
        "  scale_free final mean±std  = "
        f"{summary['scale_free']['final_mean']:.4f} ± {summary['scale_free']['final_std']:.4f}"
    )

    result = {"seeds": seeds, "runs": runs, "summary": summary}

    if args.plot:
        out_drift = _save_plot("sw_sf_bot_resilience_drift.png")
        out_basebot = _save_plot("sw_sf_baseline_vs_bot_variance.png")
        out_final = _save_plot("sw_sf_resilience_final_metrics.png")

        sw_drift = np.array([r["small_world"]["drift_series"] for r in runs], dtype=float)
        sf_drift = np.array([r["scale_free"]["drift_series"] for r in runs], dtype=float)
        sw_drift_mean = np.mean(sw_drift, axis=0)
        sf_drift_mean = np.mean(sf_drift, axis=0)

        plt.figure()
        plt.plot(sw_drift_mean, marker="o", label="small_world drift (mean)")
        plt.plot(sf_drift_mean, marker="s", label="scale_free drift (mean)")
        plt.xlabel("Timestep")
        plt.ylabel("Mean Semantic Drift (bot vs baseline)")
        plt.title(f"Small-World vs Scale-Free Resilience (Seeds: {','.join(str(s) for s in seeds)})")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.grid(True)
        plt.legend()
        plt.savefig(out_drift, dpi=150)
        print(f"Saved {out_drift}")

        sw_base = np.array([r["small_world"]["baseline_variance"] for r in runs], dtype=float)
        sw_bot = np.array([r["small_world"]["bot_variance"] for r in runs], dtype=float)
        sf_base = np.array([r["scale_free"]["baseline_variance"] for r in runs], dtype=float)
        sf_bot = np.array([r["scale_free"]["bot_variance"] for r in runs], dtype=float)

        plt.figure()
        plt.plot(np.mean(sw_base, axis=0), marker="o", label="small_world baseline (mean)")
        plt.plot(np.mean(sw_bot, axis=0), marker="x", label="small_world + bot (mean)")
        plt.plot(np.mean(sf_base, axis=0), marker="s", label="scale_free baseline (mean)")
        plt.plot(np.mean(sf_bot, axis=0), marker="d", label="scale_free + bot (mean)")
        plt.xlabel("Timestep")
        plt.ylabel("Semantic Variance")
        plt.title(f"Baseline vs Bot Variance by Topology (Seeds: {','.join(str(s) for s in seeds)})")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.grid(True)
        plt.legend()
        plt.savefig(out_basebot, dpi=150)
        print(f"Saved {out_basebot}")

        labels = ["small_world", "scale_free"]
        final_vals = [summary[k]["final_mean"] for k in labels]
        final_err = [summary[k]["final_std"] for k in labels]
        cum_vals = [summary[k]["cum_mean"] for k in labels]
        cum_err = [summary[k]["cum_std"] for k in labels]
        x = np.arange(len(labels))
        w = 0.35
        plt.figure()
        plt.bar(x - w / 2, final_vals, yerr=final_err, width=w, capsize=5, label="final drift")
        plt.bar(x + w / 2, cum_vals, yerr=cum_err, width=w, capsize=5, label="cumulative drift")
        plt.xticks(x, labels)
        plt.ylabel("Drift Metric (lower is better)")
        plt.title(f"Resilience Metrics: Small-World vs Scale-Free (Seeds: {','.join(str(s) for s in seeds)})")
        plt.grid(True)
        plt.legend()
        plt.savefig(out_final, dpi=150)
        print(f"Saved {out_final}")
    return result


def main_hub_periphery_intervention(args):
    """Compare bot-on-hub vs bot-on-periphery impact for one topology."""
    from src.intervention import run_hub_vs_periphery_bot_experiment
    from src.nodes_data import load_nodes_data

    nodes_path = PROJECT_ROOT / "src" / "nodes.json"
    nodes_data = load_nodes_data(nodes_path)
    result = run_hub_vs_periphery_bot_experiment(
        nodes_data=nodes_data,
        topic=args.topic,
        steps=args.steps,
        seed=args.seed,
        sf_m=args.sf_m,
        bot_post_prob=args.bot_prob,
        bot_post_multiplier=args.bot_mult,
        bot_target_frac=args.bot_target_frac,
    )

    print(f"\nTopology={result['topology']}, nodes={result['n_nodes']}, edges={result['n_edges']}")
    print(f"Bot target count per case={result['hub']['target_count']}")
    print(f"Hub final drift={result['hub']['final_drift']:.4f}, cumulative={result['hub']['cumulative_drift']:.4f}")
    print(
        f"Periphery final drift={result['periphery']['final_drift']:.4f}, "
        f"cumulative={result['periphery']['cumulative_drift']:.4f}"
    )

    if args.plot:
        out_drift = _save_plot("hub_vs_periphery_drift.png")
        out_var = _save_plot("hub_vs_periphery_variance.png")
        out_metric = _save_plot("hub_vs_periphery_metrics.png")

        plt.figure()
        plt.plot(result["hub"]["drift_series"], marker="o", label="bot->hub drift")
        plt.plot(result["periphery"]["drift_series"], marker="s", label="bot->periphery drift")
        plt.xlabel("Timestep")
        plt.ylabel("Mean Semantic Drift (vs baseline)")
        plt.title("Hub vs Periphery Bot Impact (scale_free)")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.grid(True)
        plt.legend()
        plt.savefig(out_drift, dpi=150)
        print(f"Saved {out_drift}")

        plt.figure()
        plt.plot(result["baseline_variance"], marker="o", label="baseline")
        plt.plot(result["hub"]["bot_variance"], marker="x", label="bot->hub")
        plt.plot(result["periphery"]["bot_variance"], marker="d", label="bot->periphery")
        plt.xlabel("Timestep")
        plt.ylabel("Semantic Variance")
        plt.title("Baseline vs Bot Placement (scale_free)")
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.grid(True)
        plt.legend()
        plt.savefig(out_var, dpi=150)
        print(f"Saved {out_var}")

        labels = ["hub", "periphery"]
        finals = [result["hub"]["final_drift"], result["periphery"]["final_drift"]]
        cums = [result["hub"]["cumulative_drift"], result["periphery"]["cumulative_drift"]]
        x = np.arange(2)
        w = 0.35
        plt.figure()
        plt.bar(x - w / 2, finals, width=w, label="final drift")
        plt.bar(x + w / 2, cums, width=w, label="cumulative drift")
        plt.xticks(x, labels)
        plt.ylabel("Drift Metric (lower is better)")
        plt.title("Hub vs Periphery Resilience Metrics (scale_free)")
        plt.grid(True)
        plt.legend()
        plt.savefig(out_metric, dpi=150)
        print(f"Saved {out_metric}")
    return result


def main_hub_periphery_multiseed(args):
    """Run hub-vs-periphery experiment across multiple seeds and aggregate stats."""
    from src.intervention import run_hub_vs_periphery_multiseed
    from src.nodes_data import load_nodes_data

    seeds = list(SEED_LIST)

    nodes_path = _nodes_path_by_dataset(args.dataset)
    nodes_data = load_nodes_data(nodes_path)
    result = run_hub_vs_periphery_multiseed(
        nodes_data=nodes_data,
        topic=args.topic,
        steps=args.steps,
        seeds=seeds,
        sf_m=args.sf_m,
        bot_post_prob=args.bot_prob,
        bot_post_multiplier=args.bot_mult,
        bot_target_frac=args.bot_target_frac,
    )

    s = result["summary"]
    print(f"\nSeeds={result['seeds']}, topology={result['topology']}")
    print(f"Hub final drift mean±std: {s['hub_final_mean']:.4f} ± {s['hub_final_std']:.4f}")
    print(f"Per final drift mean±std: {s['per_final_mean']:.4f} ± {s['per_final_std']:.4f}")
    print(f"Hub cumulative mean±std:  {s['hub_cum_mean']:.4f} ± {s['hub_cum_std']:.4f}")
    print(f"Per cumulative mean±std:  {s['per_cum_mean']:.4f} ± {s['per_cum_std']:.4f}")

    if args.plot:
        out_err = _save_plot("hub_vs_periphery_multiseed_errorbar.png")
        out_bar = _save_plot("hub_vs_periphery_multiseed_metrics.png")

        x = np.array([0, 1])
        final_means = np.array([s["hub_final_mean"], s["per_final_mean"]], dtype=float)
        final_stds = np.array([s["hub_final_std"], s["per_final_std"]], dtype=float)
        cum_means = np.array([s["hub_cum_mean"], s["per_cum_mean"]], dtype=float)
        cum_stds = np.array([s["hub_cum_std"], s["per_cum_std"]], dtype=float)

        plt.figure()
        plt.errorbar(x, final_means, yerr=final_stds, fmt="o-", capsize=5, label="final drift")
        plt.errorbar(x, cum_means, yerr=cum_stds, fmt="s-", capsize=5, label="cumulative drift")
        plt.xticks(x, ["hub", "periphery"])
        plt.ylabel("Drift metric (lower is better)")
        plt.title("Hub vs Periphery (Multi-seed) - scale_free")
        plt.grid(True)
        plt.legend()
        plt.savefig(out_err, dpi=150)
        print(f"Saved {out_err}")

        w = 0.35
        plt.figure()
        plt.bar(x - w / 2, final_means, yerr=final_stds, width=w, capsize=5, label="final drift")
        plt.bar(x + w / 2, cum_means, yerr=cum_stds, width=w, capsize=5, label="cumulative drift")
        plt.xticks(x, ["hub", "periphery"])
        plt.ylabel("Drift metric (lower is better)")
        plt.title(f"Hub vs Periphery Metrics (Seeds: {','.join(str(v) for v in seeds)})")
        plt.grid(True)
        plt.legend()
        plt.savefig(out_bar, dpi=150)
        print(f"Saved {out_bar}")

        # Also save per-seed figures to avoid manual reruns/copy.
        for run in result["runs"]:
            seed = int(run["seed"])
            out_seed_drift = _save_plot(f"hub_vs_periphery_drift_seed{seed}.png")
            out_seed_var = _save_plot(f"hub_vs_periphery_variance_seed{seed}.png")
            out_seed_metric = _save_plot(f"hub_vs_periphery_metrics_seed{seed}.png")

            plt.figure()
            plt.plot(run["hub"]["drift_series"], marker="o", label="bot->hub drift")
            plt.plot(run["periphery"]["drift_series"], marker="s", label="bot->periphery drift")
            plt.xlabel("Timestep")
            plt.ylabel("Mean Semantic Drift (vs baseline)")
            plt.title(f"Hub vs Periphery Bot Impact (scale_free, seed={seed})")
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.grid(True)
            plt.legend()
            plt.savefig(out_seed_drift, dpi=150)
            print(f"Saved {out_seed_drift}")

            plt.figure()
            plt.plot(run["baseline_variance"], marker="o", label="baseline")
            plt.plot(run["hub"]["bot_variance"], marker="x", label="bot->hub")
            plt.plot(run["periphery"]["bot_variance"], marker="d", label="bot->periphery")
            plt.xlabel("Timestep")
            plt.ylabel("Semantic Variance")
            plt.title(f"Baseline vs Bot Placement (scale_free, seed={seed})")
            plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
            plt.grid(True)
            plt.legend()
            plt.savefig(out_seed_var, dpi=150)
            print(f"Saved {out_seed_var}")

            labels = ["hub", "periphery"]
            finals = [run["hub"]["final_drift"], run["periphery"]["final_drift"]]
            cums = [run["hub"]["cumulative_drift"], run["periphery"]["cumulative_drift"]]
            x = np.arange(2)
            w = 0.35
            plt.figure()
            plt.bar(x - w / 2, finals, width=w, label="final drift")
            plt.bar(x + w / 2, cums, width=w, label="cumulative drift")
            plt.xticks(x, labels)
            plt.ylabel("Drift Metric (lower is better)")
            plt.title(f"Hub vs Periphery Resilience Metrics (scale_free, seed={seed})")
            plt.grid(True)
            plt.legend()
            plt.savefig(out_seed_metric, dpi=150)
            print(f"Saved {out_seed_metric}")
    return result


def main_hybrid(args):
    """Run fully black-box LLM updates; evaluate only at end with SBERT."""
    from src.hybrid_egt import run_hybrid_llm_egt

    print("Creating network (ER: 20 nodes)...")
    G = create_erdos_renyi_graph(n=20, p=DEFAULT_ER_EDGE_PROB, seed=args.seed)
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
    return _side_from_name_general(name)


def _group_semantic_change_series(
    nodes_data: list[dict],
    opinions_history: list[list[str]],
) -> dict[str, list[float]]:
    """
    Compute per-step mean semantic drift (vs t=0) for each political group.
    """
    from src.measurement import embed_opinions

    groups = CAMP_LABELS
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
    Camp shares over time using SBERT nearest-prototype party classification.

    For each timestep, newly generated opinions are embedded and mapped to the
    nearest persona prototype from the current dataset (nodes/senate), with labels
    resolved to party camps (democracy/republican).
    """
    camps = list(CAMP_LABELS)
    classify_fn = _build_classifier_from_nodes(nodes_data)
    n_steps = len(opinions_history)
    result = {c: [0.0] * n_steps for c in camps}
    for t in range(n_steps):
        opinions_t = opinions_history[t]
        if not opinions_t:
            continue
        labels_t = classify_fn(opinions_t)
        counts_t = {c: 0 for c in camps}
        for lb in labels_t:
            if lb in counts_t:
                counts_t[lb] += 1
        denom = len(opinions_t)
        for c in camps:
            result[c][t] = (float(counts_t.get(c, 0)) / denom) if denom > 0 else 0.0
    return result


def main_topology(args):
    """
    Run semantic simulation on ER graph using nodes.json personas.
    """
    from src.network import create_erdos_renyi_graph
    from src.nodes_data import load_nodes_data
    from src.simulation import create_agents_from_nodes_data, run_semantic

    nodes_path = PROJECT_ROOT / "src" / "nodes.json"
    nodes_data = load_nodes_data(nodes_path)
    n = len(nodes_data)
    print(f"Loaded node personas: n={n} from {nodes_path}")

    er_dir = []
    er_group_change_series = {g: [] for g in CAMP_LABELS}
    er_group_change = {g: 0.0 for g in CAMP_LABELS}
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

    print("\nPer-step direction (ER):")
    for step, label in enumerate(er_dir, start=1):
        print(f"  t={step}: {label} (variance={er_var[step]:.4f})")
    er_stable = _first_stable_step(er_var)
    print(f"\nER first stable step (|delta variance|<=1e-3): {er_stable}")

    print("\nFinal semantic change by group (ER):")
    for g in CAMP_LABELS:
        print(f"  {g}: {er_group_change[g]:.4f}")

    if args.plot:
        groups = CAMP_LABELS
        camps = list(CAMP_LABELS)
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
        plt.title("ER Camp Share Dynamics (36 persona nearest-neighbor)")
        plt.ylim(0.0, 1.0)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.grid(True)
        plt.legend()
        plt.savefig(out_er_share, dpi=150)
        print(f"Saved {out_er_share}")

    if args.animate:
        out = _save_network_animation(
            G_er,
            er_opinion_history,
            topic=args.topic,
            out_name="topology_er_network_evolution.gif",
            layout_seed=args.layout_seed,
            fps=args.anim_fps,
        )
        print(f"Saved {out}")

    return {
        "er_variance": er_var,
        "er_direction": er_dir,
        "er_group_change_series": er_group_change_series,
        "er_camp_share_series": er_camp_share_series,
        "er_group_change": er_group_change,
    }


def main_evolution_gif(args):
    """
    Generate ER evolution GIFs for selected dataset(s).
    """
    from src.network import create_erdos_renyi_graph
    from src.nodes_data import load_nodes_data
    from src.simulation import create_agents_from_nodes_data, run_semantic

    fixed_topic = FIXED_TOPIC
    seeds = list(SEED_LIST)
    all_datasets = {
        "nodes": PROJECT_ROOT / "src" / "nodes.json",
        "senate": PROJECT_ROOT / "src" / "senate_nodes.json",
    }
    if args.dataset == "both":
        datasets = list(all_datasets.items())
    else:
        datasets = [(args.dataset, all_datasets[args.dataset])]
    outputs = {}
    for seed in seeds:
        for tag, nodes_path in datasets:
            nodes_data = load_nodes_data(nodes_path)
            n = len(nodes_data)
            print(f"\n[{tag}|seed={seed}] Loaded personas: n={n} from {nodes_path}")
            G = create_erdos_renyi_graph(n=n, p=args.er_p, seed=seed)
            print(f"[{tag}|seed={seed}] ER graph: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")
            agents = create_agents_from_nodes_data(G, nodes_data, topic=fixed_topic)
            history = [[a.current_opinion for a in agents]]

            def _capture(_t, agents_state):
                history.append([a.current_opinion for a in agents_state])

            run_semantic(G, agents, fixed_topic, steps=args.steps, on_step=_capture)
            classifier = _build_classifier_from_nodes(nodes_data)
            out = _save_network_animation(
                G,
                history,
                topic=fixed_topic,
                out_name=f"topology_er_network_evolution_{tag}_seed{seed}.gif",
                classify_fn=classifier,
                layout_seed=seed,
                fps=args.anim_fps,
            )
            print(f"[{tag}|seed={seed}] Saved {out}")
            outputs[f"{tag}_seed{seed}"] = out
    return outputs


def main():
    """Parse CLI and dispatch to mode handler."""
    parser = argparse.ArgumentParser(
        description="Semantic Opinion Dynamics: LLM Agents on Complex Networks"
    )
    sub = parser.add_subparsers(dest="mode", help="Mode to run")

    p_swsf = sub.add_parser(
        "sw_sf_intervention",
        help="Compare small-world vs scale-free resilience under disinformation bot",
    )
    p_swsf.add_argument(
        "--dataset",
        choices=("nodes", "senate"),
        default="nodes",
        help="Persona dataset to run experiments on",
    )
    p_swsf.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p_swsf.add_argument("--sw-radius", type=float, default=RGG_RADIUS, help="Base RGG radius for small-world")
    p_swsf.add_argument(
        "--sw-long-prob",
        type=float,
        default=LONG_RANGE_FRACTION,
        help="Probability of random long-distance edge in small-world graph",
    )
    p_swsf.add_argument("--sf-m", type=int, default=2, help="Barabasi-Albert attachment parameter m")
    p_swsf.add_argument("--bot-prob", type=float, default=DEFAULT_BOT_POST_PROB, help="Bot posting probability")
    p_swsf.add_argument(
        "--bot-mult",
        type=int,
        default=DEFAULT_BOT_POST_MULTIPLIER,
        help="Bot repetition factor when posting",
    )
    p_swsf.add_argument(
        "--bot-connect-prob",
        type=float,
        default=DEFAULT_BOT_CONNECT_PROB,
        help="Probability a node is connected to bot",
    )
    p_swsf.add_argument("--plot", action="store_true")
    p_swsf.set_defaults(func=main_sw_sf_intervention)

    p_hub_ms = sub.add_parser(
        "hub_periphery_multiseed",
        help="Aggregate scale-free hub vs periphery experiment across multiple seeds",
    )
    p_hub_ms.add_argument(
        "--dataset",
        choices=("nodes", "senate"),
        default="nodes",
        help="Persona dataset to run experiments on",
    )
    p_hub_ms.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p_hub_ms.add_argument("--sf-m", type=int, default=2, help="BA attachment parameter for scale-free topology")
    p_hub_ms.add_argument("--bot-prob", type=float, default=DEFAULT_BOT_POST_PROB, help="Bot posting probability")
    p_hub_ms.add_argument(
        "--bot-mult",
        type=int,
        default=DEFAULT_BOT_POST_MULTIPLIER,
        help="Bot repetition factor when posting",
    )
    p_hub_ms.add_argument(
        "--bot-target-frac",
        type=float,
        default=DEFAULT_BOT_TARGET_FRAC,
        help="Fraction of nodes directly connected to bot in each case",
    )
    p_hub_ms.add_argument(
        "--plot",
        action="store_true",
    )
    p_hub_ms.set_defaults(func=main_hub_periphery_multiseed)

    p_evo = sub.add_parser(
        "evolution_gif",
        help="Generate ER evolution GIFs for nodes/senate datasets",
    )
    p_evo.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    p_evo.add_argument("--er-p", type=float, default=DEFAULT_ER_EDGE_PROB, help="Edge probability for ER graph")
    p_evo.add_argument(
        "--dataset",
        choices=("nodes", "senate", "both"),
        default="both",
        help="Which dataset to render as GIF(s)",
    )
    p_evo.add_argument("--anim-fps", type=int, default=2, help="FPS for GIF animation")
    p_evo.set_defaults(func=main_evolution_gif)

    args = parser.parse_args()
    if args.mode is None:
        parser.print_help()
        return 1
    # Force one canonical topic across all experiments.
    args.topic = FIXED_TOPIC
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
