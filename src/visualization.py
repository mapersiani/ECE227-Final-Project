"""
Visualization module for opinion dynamics experiments.

Contains all plotting functions extracted from main.py.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import Normalize

from src.config import DEFAULT_SEED, PERSONA_BLOCKS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def node_positions(G: nx.Graph, seed: int = DEFAULT_SEED) -> dict[int, tuple[float, float]]:
    """Get node layout: use stored positions if available, else spring layout."""
    has_pos = all("pos" in G.nodes[n] for n in G.nodes())
    if has_pos:
        return {int(n): tuple(G.nodes[n]["pos"]) for n in G.nodes()}
    return nx.spring_layout(G, seed=seed)


def opinion_drift_by_node(agents) -> dict[int, float]:
    """Compute per-node opinion drift (cosine distance from initial to final)."""
    from src.measurement import embed_opinions

    if not agents:
        return {}
    initial = [a.initial_opinion for a in agents]
    final = [a.current_opinion for a in agents]
    emb_i = embed_opinions(initial)
    emb_f = embed_opinions(final)

    norm_i = np.linalg.norm(emb_i, axis=1, keepdims=True)
    norm_f = np.linalg.norm(emb_f, axis=1, keepdims=True)
    cos_sim = np.sum(
        (emb_i / np.where(norm_i == 0, 1e-9, norm_i)) * (emb_f / np.where(norm_f == 0, 1e-9, norm_f)),
        axis=1,
    )
    drift = 1.0 - np.clip(cos_sim, -1.0, 1.0)
    return {int(a.node_id): float(d) for a, d in zip(agents, drift)}


def condition_label(cond: tuple[str, str, str]) -> str:
    graph, persona_set, bot = cond
    return f"{graph} | {persona_set} | bot={bot}"


def _side_entropy(democrat: int, republican: int, independent: int) -> float:
    counts = np.array([democrat, republican, independent], dtype=float)
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    probs = counts / total
    nonzero = probs[probs > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


# ---------------------------------------------------------------------------
# Single-run plots
# ---------------------------------------------------------------------------

def plot_topology(
    G: nx.Graph,
    out_path: Path,
    title: str,
    seed: int = DEFAULT_SEED,
) -> None:
    pos = node_positions(G, seed=seed)

    side_colors = {
        "democrat": "#3b82f6",
        "republican": "#ef4444",
        "independent": "#22c55e",
        "bot": "#111827",
        "unknown": "#9ca3af",
    }
    node_colors = [side_colors.get(str(G.nodes[n].get("side", "unknown")), "#9ca3af") for n in G.nodes()]

    has_edge_type = any("edge_type" in d for _, _, d in G.edges(data=True))
    local_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") != "long_range"]
    long_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "long_range"]

    plt.figure(figsize=(12, 8))
    nx.draw_networkx_edges(G, pos, edgelist=local_edges, alpha=0.28, edge_color="#9ca3af", width=1.0)
    if has_edge_type and long_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=long_edges, alpha=0.6, edge_color="#8b5cf6", style="dashed", width=1.3,
        )
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=230, linewidths=0.5, edgecolors="white")

    deg = dict(G.degree())
    label_nodes = [n for n, _ in sorted(deg.items(), key=lambda x: x[1], reverse=True)[:8]]
    labels = {n: str(G.nodes[n].get("name", n)).split("_")[-1] for n in label_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=7)

    plt.title(title)
    plt.xlabel("Ideological Position")
    plt.ylabel("Engagement Level")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_drift_network(
    G: nx.Graph,
    agents,
    out_path: Path,
    title: str,
    seed: int = DEFAULT_SEED,
) -> None:
    drift = opinion_drift_by_node(agents)
    if not drift:
        return

    pos = node_positions(G, seed=seed)
    nodes = list(G.nodes())
    vals = np.array([float(drift.get(int(n), 0.0)) for n in nodes], dtype=float)

    cmap = plt.cm.plasma
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    denom = (vmax - vmin) if (vmax > vmin) else 1.0
    colors = [cmap((v - vmin) / denom) for v in vals]

    local_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") != "long_range"]
    long_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "long_range"]

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_edges(G, pos, edgelist=local_edges, alpha=0.22, edge_color="#9ca3af", width=1.0, ax=ax)
    if long_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=long_edges, alpha=0.58, edge_color="#8b5cf6", style="dashed", width=1.2, ax=ax,
        )
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=250, linewidths=0.5, edgecolors="white", ax=ax)

    top_nodes = [n for n, _ in sorted(drift.items(), key=lambda x: x[1], reverse=True)[:10]]
    labels = {int(n): str(G.nodes[int(n)].get("name", n)).split("_")[-1] for n in top_nodes if int(n) in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Opinion Drift (cosine Δ from initial)")

    ax.set_title(title)
    ax.set_xlabel("Ideological Position")
    ax.set_ylabel("Engagement Level")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close()


def plot_vote_comparison(
    initial_votes: dict[str, int],
    final_votes: dict[str, int],
    out_path: Path,
    title: str,
) -> None:
    labels = ["SUPPORT", "AGAINST", "ABSTAIN"]
    init_counts = [initial_votes.get(l, 0) for l in labels]
    final_counts = [final_votes.get(l, 0) for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - width / 2, init_counts, width, label="Initial Vote", color="#60a5fa")
    rects2 = ax.bar(x + width / 2, final_counts, width, label="Final Vote", color="#f472b6")

    ax.set_ylabel("Number of Nodes")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()


def plot_single_series(
    data: list[float],
    out_path: Path,
    title: str,
    ylabel: str,
    marker: str = "o",
    color: str | None = None,
) -> None:
    """Generic single time-series plot (variance, polarization, drift, etc.)."""
    plt.figure()
    kwargs: dict = {"marker": marker}
    if color:
        kwargs["color"] = color
    plt.plot(data, **kwargs)
    plt.xlabel("Timestep")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_side_counts(
    side_counts: list[dict[str, int]],
    out_path: Path,
    title: str,
) -> None:
    steps = list(range(len(side_counts)))
    labels = [s for s in PERSONA_BLOCKS if s in side_counts[0]]
    plt.figure()
    for lab in labels:
        series = [c[lab] for c in side_counts]
        plt.plot(steps, series, marker="o", label=lab)
    plt.xlabel("Timestep")
    plt.ylabel("Count of agents")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path, dpi=160)
    plt.close()


# ---------------------------------------------------------------------------
# Matrix plots
# ---------------------------------------------------------------------------

def plot_matrix_condition_lines(rows: list[dict[str, object]], out_path: Path) -> None:
    by_condition: dict[tuple[str, str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cond = (str(row["graph"]), str(row["persona_set"]), str(row["bot"]))
        t = int(row["t"])
        by_condition[cond][t].append(float(row["variance"]))

    plt.figure(figsize=(11, 6))
    for cond in sorted(by_condition.keys()):
        ts = sorted(by_condition[cond].keys())
        means = [float(np.mean(by_condition[cond][t])) for t in ts]
        label = condition_label(cond)
        plt.plot(ts, means, marker="o", linewidth=2, label=label)
    plt.xlabel("Timestep")
    plt.ylabel("Mean Variance Across Seeds")
    plt.title("Matrix: Condition Variance Trajectories")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_matrix_final_step_bars(rows: list[dict[str, object]], out_path: Path, final_t: int) -> None:
    by_condition: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if int(row["t"]) != final_t:
            continue
        cond = (str(row["graph"]), str(row["persona_set"]), str(row["bot"]))
        by_condition[cond].append(float(row["variance"]))

    conds = sorted(by_condition.keys())
    means = [float(np.mean(by_condition[c])) for c in conds]
    stds = [float(np.std(by_condition[c])) for c in conds]
    labels = [condition_label(c) for c in conds]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(conds))
    plt.bar(x, means, yerr=stds, capsize=5, color="#60a5fa", edgecolor="#1e3a8a", alpha=0.85)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel(f"Variance at t={final_t}")
    plt.title("Matrix: Final-Step Variance (mean +/- std across seeds)")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_matrix_variance_heatmap(rows: list[dict[str, object]], out_path: Path) -> None:
    by_condition_t: dict[tuple[str, str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        cond = (str(row["graph"]), str(row["persona_set"]), str(row["bot"]))
        t = int(row["t"])
        by_condition_t[cond][t].append(float(row["variance"]))

    conds = sorted(by_condition_t.keys())
    ts = sorted({int(row["t"]) for row in rows})
    mat = np.zeros((len(conds), len(ts)), dtype=float)

    for i, cond in enumerate(conds):
        for j, t in enumerate(ts):
            vals = by_condition_t[cond].get(t, [])
            mat[i, j] = float(np.mean(vals)) if vals else np.nan

    plt.figure(figsize=(12, 6))
    im = plt.imshow(mat, aspect="auto", cmap="viridis")
    plt.colorbar(im, label="Mean Variance Across Seeds")
    plt.yticks(np.arange(len(conds)), [condition_label(c) for c in conds])
    plt.xticks(np.arange(len(ts)), ts)
    plt.xlabel("Timestep")
    plt.ylabel("Condition")
    plt.title("Matrix: Variance Heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_matrix_bot_effect(rows: list[dict[str, object]], out_path: Path) -> None:
    series: dict[tuple[str, str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        graph = str(row["graph"])
        persona_set = str(row["persona_set"])
        bot = str(row["bot"])
        if str(row["model"]) != "semantic":
            continue
        t = int(row["t"])
        key = (graph, persona_set, bot)
        series[key][t].append(float(row["variance"]))

    graph_persona = sorted({(g, p) for g, p, _ in series.keys()})
    plt.figure(figsize=(11, 6))
    for graph, persona_set in graph_persona:
        on = series.get((graph, persona_set, "on"), {})
        off = series.get((graph, persona_set, "off"), {})
        ts = sorted(set(on.keys()) & set(off.keys()))
        if not ts:
            continue
        effects: list[float] = []
        for t in ts:
            mean_on = float(np.mean(on[t]))
            mean_off = float(np.mean(off[t]))
            effects.append(mean_on - mean_off)
        plt.plot(ts, effects, marker="o", linewidth=2, label=f"{graph} | {persona_set}: bot on - off")

    plt.axhline(0.0, color="#374151", linewidth=1.0, linestyle="--")
    plt.xlabel("Timestep")
    plt.ylabel("Variance Difference")
    plt.title("Matrix: Bot Effect Over Time")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_matrix_side_counts(rows: list[dict[str, object]], out_path: Path) -> None:
    """Mean democrat/republican/independent counts over time by condition."""
    by_cond_t: dict[tuple[str, str, str], dict[int, list[tuple[int, int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if str(row["model"]) != "semantic":
            continue
        if row["democrat_count"] is None:
            continue
        cond = (str(row["graph"]), str(row["persona_set"]), str(row["bot"]))
        t = int(row["t"])
        by_cond_t[cond][t].append((
            int(row["democrat_count"]),
            int(row["republican_count"]),
            int(row["independent_count"]),
        ))
    conds = sorted(by_cond_t.keys())
    ncols = 4
    nrows = (len(conds) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    colors = {"democrat": "#3b82f6", "republican": "#ef4444", "independent": "#22c55e"}
    for idx, cond in enumerate(conds):
        ax = axes_flat[idx]
        ts = sorted(by_cond_t[cond].keys())
        dem = [float(np.mean([x[0] for x in by_cond_t[cond][t]])) for t in ts]
        rep = [float(np.mean([x[1] for x in by_cond_t[cond][t]])) for t in ts]
        ind = [float(np.mean([x[2] for x in by_cond_t[cond][t]])) for t in ts]
        ax.plot(ts, dem, marker="o", markersize=4, color=colors["democrat"], label="democrat")
        ax.plot(ts, rep, marker="o", markersize=4, color=colors["republican"], label="republican")
        ax.plot(ts, ind, marker="o", markersize=4, color=colors["independent"], label="independent")
        ax.set_title(condition_label(cond))
        ax.set_xlabel("t")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    for j in range(idx + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Matrix: Side Counts Over Time by Condition", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_matrix_side_entropy(rows: list[dict[str, object]], out_path: Path) -> None:
    by_condition_t: dict[tuple[str, str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if str(row["model"]) != "semantic":
            continue
        if row["democrat_count"] is None:
            continue
        graph = str(row["graph"])
        persona_set = str(row["persona_set"])
        bot = str(row["bot"])
        t = int(row["t"])
        entropy = _side_entropy(
            int(row["democrat_count"]),
            int(row["republican_count"]),
            int(row["independent_count"]),
        )
        by_condition_t[(graph, persona_set, bot)][t].append(entropy)

    plt.figure(figsize=(11, 6))
    for cond in sorted(by_condition_t.keys()):
        ts = sorted(by_condition_t[cond].keys())
        means = [float(np.mean(by_condition_t[cond][t])) for t in ts]
        label = condition_label(cond)
        plt.plot(ts, means, marker="o", linewidth=2, label=label)

    plt.xlabel("Timestep")
    plt.ylabel("Side Entropy (bits)")
    plt.title("Matrix: Semantic Side-Mix Entropy")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_side_transition_timing_bot_on_off(
    changed_counts_by_bot: dict[str, np.ndarray],
    total_counts_by_bot: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    plt.figure(figsize=(11, 6))
    for bot in ("off", "on"):
        changed = changed_counts_by_bot.get(bot)
        total = total_counts_by_bot.get(bot)
        if changed is None or total is None or len(changed) == 0:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            rates = np.divide(changed, total, where=total > 0) * 100.0
        ts = np.arange(1, len(rates) + 1)
        plt.plot(ts, rates, marker="o", linewidth=2, label=f"bot={bot}")

    plt.xlabel("Transition Step (t -> t+1)")
    plt.ylabel("Agents Changing Side (%)")
    plt.title("Matrix: Side Transition Timing by Bot Condition")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_side_final_transition_matrix_bot_on_off(
    final_transitions_by_bot: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    bots = ("off", "on")
    mats = [
        final_transitions_by_bot.get(bot, np.zeros((len(PERSONA_BLOCKS), len(PERSONA_BLOCKS))))
        for bot in bots
    ]
    norm_mats: list[np.ndarray] = []
    for mat in mats:
        row_sum = mat.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = np.divide(mat, row_sum, where=row_sum > 0)
        norm_mats.append(norm)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    vmax = max(float(np.max(norm)) for norm in norm_mats) if norm_mats else 1.0
    vmax = max(vmax, 1e-6)
    im = None
    for ax, bot, raw_mat, norm_mat in zip(axes, bots, mats, norm_mats):
        im = ax.imshow(norm_mat, cmap="YlOrRd", vmin=0.0, vmax=vmax)
        ax.set_title(f"bot={bot}")
        ax.set_xlabel("Final side (t=final)")
        ax.set_ylabel("Initial side (t=0)")
        ax.set_xticks(np.arange(len(PERSONA_BLOCKS)))
        ax.set_yticks(np.arange(len(PERSONA_BLOCKS)))
        ax.set_xticklabels(PERSONA_BLOCKS, rotation=25, ha="right")
        ax.set_yticklabels(PERSONA_BLOCKS)
        for i in range(len(PERSONA_BLOCKS)):
            for j in range(len(PERSONA_BLOCKS)):
                pct = norm_mat[i, j] * 100.0
                cnt = int(raw_mat[i, j])
                text_color = "white" if norm_mat[i, j] > 0.55 * vmax else "#111827"
                ax.text(j, i, f"{pct:.0f}%\n(n={cnt})", ha="center", va="center", fontsize=8, color=text_color)

    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.85, label="Probability from Initial Side")
    fig.suptitle("Matrix: Initial -> Final Side Transition by Bot Condition", fontsize=13)
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_side_transition_matrix_bot_on_off(
    transitions_by_bot: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    bots = ("off", "on")
    mats = [transitions_by_bot.get(bot, np.zeros((len(PERSONA_BLOCKS), len(PERSONA_BLOCKS)))) for bot in bots]
    norm_mats: list[np.ndarray] = []
    for mat in mats:
        row_sum = mat.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = np.divide(mat, row_sum, where=row_sum > 0)
        norm_mats.append(norm)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    vmax = max(float(np.max(norm)) for norm in norm_mats) if norm_mats else 1.0
    vmax = max(vmax, 1e-6)
    im = None
    for ax, bot, raw_mat, norm_mat in zip(axes, bots, mats, norm_mats):
        im = ax.imshow(norm_mat, cmap="Blues", vmin=0.0, vmax=vmax)
        ax.set_title(f"bot={bot}")
        ax.set_xlabel("To side (t+1)")
        ax.set_ylabel("From side (t)")
        ax.set_xticks(np.arange(len(PERSONA_BLOCKS)))
        ax.set_yticks(np.arange(len(PERSONA_BLOCKS)))
        ax.set_xticklabels(PERSONA_BLOCKS, rotation=25, ha="right")
        ax.set_yticklabels(PERSONA_BLOCKS)
        for i in range(len(PERSONA_BLOCKS)):
            for j in range(len(PERSONA_BLOCKS)):
                pct = norm_mat[i, j] * 100.0
                cnt = int(raw_mat[i, j])
                text_color = "white" if norm_mat[i, j] > 0.55 * vmax else "#111827"
                ax.text(j, i, f"{pct:.0f}%\n(n={cnt})", ha="center", va="center", fontsize=8, color=text_color)

    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.85, label="Transition Probability")
    fig.suptitle("Matrix: Side Transition (t -> t+1) by Bot Condition", fontsize=13)
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_matrix_polarization_trajectories(rows: list[dict[str, object]], out_path: Path) -> None:
    """Mean opinion polarization over time by condition (semantic only)."""
    by_condition: dict[tuple[str, str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if str(row.get("model", "")) != "semantic":
            continue
        if row.get("polarization") is None:
            continue
        cond = (str(row["graph"]), str(row["persona_set"]), str(row["bot"]))
        t = int(row["t"])
        by_condition[cond][t].append(float(row["polarization"]))

    plt.figure(figsize=(11, 6))
    for cond in sorted(by_condition.keys()):
        ts = sorted(by_condition[cond].keys())
        means = [float(np.mean(by_condition[cond][t])) for t in ts]
        label = condition_label(cond)
        plt.plot(ts, means, marker="s", linewidth=2, label=label)
    plt.xlabel("Timestep")
    plt.ylabel("Opinion Polarization (between/within block ratio)")
    plt.title("Matrix: Opinion Polarization Trajectories")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_matrix_drift_trajectories(rows: list[dict[str, object]], out_path: Path) -> None:
    """Mean persona drift over time by condition (semantic only)."""
    by_condition: dict[tuple[str, str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if str(row.get("model", "")) != "semantic":
            continue
        if row.get("persona_drift_mean") is None:
            continue
        cond = (str(row["graph"]), str(row["persona_set"]), str(row["bot"]))
        t = int(row["t"])
        by_condition[cond][t].append(float(row["persona_drift_mean"]))

    plt.figure(figsize=(11, 6))
    for cond in sorted(by_condition.keys()):
        ts = sorted(by_condition[cond].keys())
        means = [float(np.mean(by_condition[cond][t])) for t in ts]
        label = condition_label(cond)
        plt.plot(ts, means, marker="^", linewidth=2, label=label)
    plt.xlabel("Timestep")
    plt.ylabel("Mean Persona Drift (1 − cosine similarity)")
    plt.title("Matrix: Persona Drift Trajectories")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_matrix_vote_shift(rows: list[dict[str, object]], out_path: Path, final_t: int) -> None:
    """Grouped bar chart: initial vs. final SUPPORT/AGAINST/ABSTAIN votes by condition."""
    initial: dict[tuple[str, str, str], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    final: dict[tuple[str, str, str], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        if str(row.get("model", "")) != "semantic":
            continue
        t = int(row["t"])
        cond = (str(row["graph"]), str(row["persona_set"]), str(row["bot"]))
        if t == 0 and row.get("vote_support_count") is not None:
            for vk in ("vote_support_count", "vote_against_count", "vote_abstain_count"):
                initial[cond][vk].append(int(float(row[vk])))
        elif t == final_t and row.get("vote_support_count") is not None:
            for vk in ("vote_support_count", "vote_against_count", "vote_abstain_count"):
                final[cond][vk].append(int(float(row[vk])))

    conds = sorted(set(initial.keys()) | set(final.keys()))
    if not conds:
        return

    vote_keys = ["vote_support_count", "vote_against_count", "vote_abstain_count"]
    short = {"vote_support_count": "SUPPORT", "vote_against_count": "AGAINST", "vote_abstain_count": "ABSTAIN"}
    colors_init = {"vote_support_count": "#93c5fd", "vote_against_count": "#fca5a5", "vote_abstain_count": "#d1d5db"}
    colors_final = {"vote_support_count": "#2563eb", "vote_against_count": "#dc2626", "vote_abstain_count": "#6b7280"}

    n_conds = len(conds)
    fig, axes = plt.subplots(1, n_conds, figsize=(5 * n_conds, 6), squeeze=False)
    axes_flat = axes.flatten()

    for idx, cond in enumerate(conds):
        ax = axes_flat[idx]
        x = np.arange(len(vote_keys))
        width = 0.35
        init_vals = [int(np.mean(initial[cond][vk])) if initial[cond].get(vk) else 0 for vk in vote_keys]
        fin_vals = [int(np.mean(final[cond][vk])) if final[cond].get(vk) else 0 for vk in vote_keys]
        rects1 = ax.bar(x - width / 2, init_vals, width, label="Initial", color=[colors_init[k] for k in vote_keys])
        rects2 = ax.bar(x + width / 2, fin_vals, width, label="Final", color=[colors_final[k] for k in vote_keys])
        ax.bar_label(rects1, padding=2, fontsize=7)
        ax.bar_label(rects2, padding=2, fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([short[k] for k in vote_keys])
        ax.set_title(condition_label(cond), fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Matrix: Initial vs. Final Votes by Condition", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_matrix_analysis_pack(
    rows: list[dict[str, object]],
    out_dir: Path,
    final_t: int,
    transitions_by_persona_by_bot: dict[str, dict[str, np.ndarray]],
    final_transitions_by_persona_by_bot: dict[str, dict[str, np.ndarray]],
    changed_counts_by_persona_by_bot: dict[str, dict[str, np.ndarray]],
    total_counts_by_persona_by_bot: dict[str, dict[str, np.ndarray]],
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = [
        out_dir / "final_step_variance_bars.png",
        out_dir / "variance_heatmap.png",
        out_dir / "bot_effect_over_time.png",
        out_dir / "side_counts_over_time.png",
        out_dir / "semantic_side_entropy.png",
        out_dir / "polarization_trajectories.png",
        out_dir / "drift_trajectories.png",
        out_dir / "vote_shift_comparison.png",
    ]
    plot_matrix_final_step_bars(rows, files[0], final_t=final_t)
    plot_matrix_variance_heatmap(rows, files[1])
    plot_matrix_bot_effect(rows, files[2])
    plot_matrix_side_counts(rows, files[3])
    plot_matrix_side_entropy(rows, files[4])
    plot_matrix_polarization_trajectories(rows, files[5])
    plot_matrix_drift_trajectories(rows, files[6])
    plot_matrix_vote_shift(rows, files[7], final_t=final_t)
    for persona_set in sorted(transitions_by_persona_by_bot.keys()):
        t_by_bot = transitions_by_persona_by_bot[persona_set]
        f_by_bot = final_transitions_by_persona_by_bot[persona_set]
        c_by_bot = changed_counts_by_persona_by_bot[persona_set]
        ttot_by_bot = total_counts_by_persona_by_bot[persona_set]
        files.append(out_dir / f"side_transition_matrix_bot_on_off_{persona_set}.png")
        plot_side_transition_matrix_bot_on_off(t_by_bot, files[-1])
        files.append(out_dir / f"side_transition_timing_bot_on_off_{persona_set}.png")
        plot_side_transition_timing_bot_on_off(c_by_bot, ttot_by_bot, files[-1])
        files.append(out_dir / f"side_transition_matrix_initial_to_final_bot_on_off_{persona_set}.png")
        plot_side_final_transition_matrix_bot_on_off(f_by_bot, files[-1])
    return files

