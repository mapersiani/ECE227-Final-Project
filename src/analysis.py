"""
Analysis module for opinion dynamics experiments.

Contains data processing, graph metrics computation, and CSV writing
logic extracted from main.py.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np

from src.config import PERSONA_BLOCKS


# ---------------------------------------------------------------------------
# Graph structure metrics
# ---------------------------------------------------------------------------

def graph_structure_metrics(G: nx.Graph) -> dict[str, float | int]:
    """Compute comprehensive structural metrics for a graph."""
    n_nodes = int(G.number_of_nodes())
    n_edges = int(G.number_of_edges())
    blocks = [G.nodes[i].get("block", 0) for i in range(n_nodes)]

    degs = np.array([d for _, d in G.degree()], dtype=float)
    avg_degree = float(np.mean(degs)) if degs.size else 0.0
    min_degree = int(np.min(degs)) if degs.size else 0
    max_degree = int(np.max(degs)) if degs.size else 0
    isolates = int(np.sum(degs == 0)) if degs.size else 0

    density = float(nx.density(G)) if n_nodes > 1 else 0.0
    components = int(nx.number_connected_components(G)) if n_nodes > 0 else 0

    has_edge_type = any("edge_type" in d for _, _, d in G.edges(data=True))
    if has_edge_type:
        long_range_edges = int(sum(1 for _, _, d in G.edges(data=True) if d.get("edge_type") == "long_range"))
        local_edges = n_edges - long_range_edges
    else:
        long_range_edges = 0
        local_edges = n_edges

    # Structural
    avg_clustering = nx.average_clustering(G)
    transitivity = nx.transitivity(G)

    try:
        avg_sp = nx.average_shortest_path_length(G)
        diameter = nx.diameter(G)
    except nx.NetworkXError:
        if n_nodes > 0:
            lcc = max(nx.connected_components(G), key=len)
            H = G.subgraph(lcc)
            avg_sp = nx.average_shortest_path_length(H)
            diameter = nx.diameter(H)
        else:
            avg_sp = 0.0
            diameter = 0

    try:
        deg_assort = nx.degree_assortativity_coefficient(G)
    except Exception:
        deg_assort = float("nan")

    # Modularity using block partition
    block_communities: dict[int, set] = {}
    for i, b in enumerate(blocks):
        block_communities.setdefault(b, set()).add(i)
    communities = list(block_communities.values())
    try:
        modularity = nx.community.modularity(G, communities)
    except Exception:
        modularity = float("nan")

    # Bridge edges
    bridge_edges = set(nx.bridges(G))
    bridge_fraction = len(bridge_edges) / max(G.number_of_edges(), 1)

    # Centrality
    betweenness = nx.betweenness_centrality(G, normalized=True)
    try:
        eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        eigenvector = {i: 0.0 for i in G.nodes()}

    avg_betweenness = float(np.mean(list(betweenness.values()))) if betweenness else 0.0
    avg_eigenvector = float(np.mean(list(eigenvector.values()))) if eigenvector else 0.0

    return {
        "nodes": n_nodes,
        "edges": n_edges,
        "density": density,
        "components": components,
        "isolates": isolates,
        "avg_degree": avg_degree,
        "min_degree": min_degree,
        "max_degree": max_degree,
        "local_edges": local_edges,
        "long_range_edges": long_range_edges,
        "avg_clustering": float(avg_clustering),
        "transitivity": float(transitivity),
        "avg_shortest_path": float(avg_sp),
        "diameter": int(diameter),
        "degree_assortativity": float(deg_assort),
        "modularity": float(modularity),
        "bridge_edge_fraction": float(bridge_fraction),
        "avg_betweenness": avg_betweenness,
        "avg_eigenvector": avg_eigenvector,
    }


# ---------------------------------------------------------------------------
# Timeseries CSV (single run)
# ---------------------------------------------------------------------------

def write_run_timeseries_csv(
    out_path: Path,
    semantic_var: Optional[list[float]],
    side_counts: Optional[list[dict[str, int]]],
) -> None:
    if semantic_var is None or len(semantic_var) == 0:
        return

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "t",
                "semantic_variance",
                "democrat_count",
                "republican_count",
                "independent_count",
            ],
        )
        writer.writeheader()
        for t in range(len(semantic_var)):
            counts = side_counts[t] if side_counts is not None and t < len(side_counts) else {}
            writer.writerow(
                {
                    "t": t,
                    "semantic_variance": semantic_var[t],
                    "democrat_count": counts.get("democrat"),
                    "republican_count": counts.get("republican"),
                    "independent_count": counts.get("independent"),
                }
            )


# ---------------------------------------------------------------------------
# Matrix CSV + summary
# ---------------------------------------------------------------------------

MATRIX_CSV_FIELDNAMES = [
    "matrix_id",
    "graph",
    "persona_set",
    "model",
    "bot",
    "seed",
    "t",
    "topic",
    "variance",
    "polarization",
    "persona_drift_mean",
    "delta_from_t0",
    "delta_from_prev",
    "democrat_count",
    "republican_count",
    "independent_count",
    "measured_agents",
    "vote_support_count",
    "vote_against_count",
    "vote_abstain_count",
    "graph_nodes",
    "graph_edges",
    "graph_density",
    "graph_components",
    "graph_isolates",
    "graph_avg_degree",
    "graph_min_degree",
    "graph_max_degree",
    "graph_local_edges",
    "graph_long_range_edges",
    "graph_avg_clustering",
    "graph_transitivity",
    "graph_avg_shortest_path",
    "graph_diameter",
    "graph_degree_assortativity",
    "graph_modularity",
    "graph_bridge_edge_fraction",
    "graph_avg_betweenness",
    "graph_avg_eigenvector",
    "bot_degree",
]


def append_matrix_rows(
    rows: list[dict[str, object]],
    *,
    matrix_id: str,
    graph: str,
    persona_set: str,
    model: str,
    bot: str,
    seed: int,
    steps: int,
    topic: str,
    variances: list[float],
    polarizations: list[float] | None,
    drifts: list[float] | None,
    side_counts: Optional[list[dict[str, int]]],
    initial_votes: dict[str, int] | None,
    final_votes: dict[str, int] | None,
    graph_metrics: dict[str, float | int],
    bot_degree: Optional[int],
) -> None:
    if len(variances) != steps + 1:
        raise ValueError(
            f"Expected {steps + 1} variance points but got {len(variances)} "
            f"for {graph}/{persona_set}/{model}/{bot}/seed={seed}."
        )

    v0 = float(variances[0])
    prev: Optional[float] = None

    for t, variance in enumerate(variances):
        counts = side_counts[t] if side_counts is not None else None
        democrat = counts.get("democrat", 0) if counts else None
        republican = counts.get("republican", 0) if counts else None
        independent = counts.get("independent", 0) if counts else None
        measured_agents = (democrat + republican + independent) if counts else None

        votes_dict = None
        if t == 0 and initial_votes is not None:
            votes_dict = initial_votes
        elif t == steps and final_votes is not None:
            votes_dict = final_votes

        row = {
            "matrix_id": matrix_id,
            "graph": graph,
            "persona_set": persona_set,
            "model": model,
            "bot": bot,
            "seed": seed,
            "t": t,
            "topic": topic,
            "variance": float(variance),
            "polarization": float(polarizations[t]) if polarizations else None,
            "persona_drift_mean": float(drifts[t]) if drifts else None,
            "delta_from_t0": float(variance) - v0,
            "delta_from_prev": None if prev is None else float(variance) - prev,
            "democrat_count": democrat,
            "republican_count": republican,
            "independent_count": independent,
            "measured_agents": measured_agents,
            "vote_support_count": votes_dict.get("SUPPORT", 0) if votes_dict else None,
            "vote_against_count": votes_dict.get("AGAINST", 0) if votes_dict else None,
            "vote_abstain_count": votes_dict.get("ABSTAIN", 0) if votes_dict else None,
            "graph_nodes": graph_metrics["nodes"],
            "graph_edges": graph_metrics["edges"],
            "graph_density": graph_metrics["density"],
            "graph_components": graph_metrics["components"],
            "graph_isolates": graph_metrics["isolates"],
            "graph_avg_degree": graph_metrics["avg_degree"],
            "graph_min_degree": graph_metrics["min_degree"],
            "graph_max_degree": graph_metrics["max_degree"],
            "graph_local_edges": graph_metrics["local_edges"],
            "graph_long_range_edges": graph_metrics["long_range_edges"],
            "graph_avg_clustering": graph_metrics.get("avg_clustering"),
            "graph_transitivity": graph_metrics.get("transitivity"),
            "graph_avg_shortest_path": graph_metrics.get("avg_shortest_path"),
            "graph_diameter": graph_metrics.get("diameter"),
            "graph_degree_assortativity": graph_metrics.get("degree_assortativity"),
            "graph_modularity": graph_metrics.get("modularity"),
            "graph_bridge_edge_fraction": graph_metrics.get("bridge_edge_fraction"),
            "graph_avg_betweenness": graph_metrics.get("avg_betweenness"),
            "graph_avg_eigenvector": graph_metrics.get("avg_eigenvector"),
            "bot_degree": bot_degree,
        }
        rows.append(row)
        prev = float(variance)


def write_matrix_csv(rows: list[dict[str, object]], out_path: Path) -> None:
    if not rows:
        raise ValueError("No matrix rows generated.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=MATRIX_CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def print_matrix_summary(rows: list[dict[str, object]], final_t: int) -> dict[str, dict[str, float | int]]:
    buckets: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        if int(row["t"]) != final_t:
            continue
        if str(row.get("model", "")) != "semantic":
            continue
        key = (str(row["graph"]), str(row["persona_set"]), str(row["bot"]))
        buckets[key].append(float(row["variance"]))

    summary: dict[str, dict[str, float | int]] = {}
    print("\nFinal-step variance summary (mean +/- std across seeds):")
    for key in sorted(buckets.keys()):
        values = np.array(buckets[key], dtype=float)
        k = f"{key[0]}|{key[1]}|{key[2]}"
        summary[k] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "n": int(len(values)),
        }
        print(
            f"  {key[0]} | {key[1]} | bot={key[2]} -> "
            f"{float(np.mean(values)):.4f} +/- {float(np.std(values)):.4f} (n={len(values)})"
        )
    return summary


# ---------------------------------------------------------------------------
# Side transition accumulators
# ---------------------------------------------------------------------------

def accumulate_side_transitions(
    side_labels_over_time: list[list[str]],
    transition_counts: np.ndarray,
) -> None:
    side_index = {side: idx for idx, side in enumerate(PERSONA_BLOCKS)}
    for t in range(len(side_labels_over_time) - 1):
        src_labels = side_labels_over_time[t]
        dst_labels = side_labels_over_time[t + 1]
        n = min(len(src_labels), len(dst_labels))
        for i in range(n):
            src = src_labels[i]
            dst = dst_labels[i]
            if src in side_index and dst in side_index:
                transition_counts[side_index[src], side_index[dst]] += 1.0


def accumulate_final_side_transitions(
    side_labels_over_time: list[list[str]],
    transition_counts: np.ndarray,
) -> None:
    if len(side_labels_over_time) < 2:
        return
    side_index = {side: idx for idx, side in enumerate(PERSONA_BLOCKS)}
    src_labels = side_labels_over_time[0]
    dst_labels = side_labels_over_time[-1]
    n = min(len(src_labels), len(dst_labels))
    for i in range(n):
        src = src_labels[i]
        dst = dst_labels[i]
        if src in side_index and dst in side_index:
            transition_counts[side_index[src], side_index[dst]] += 1.0


def accumulate_transition_timing(
    side_labels_over_time: list[list[str]],
    changed_counts: np.ndarray,
    total_counts: np.ndarray,
) -> None:
    side_index = {side: idx for idx, side in enumerate(PERSONA_BLOCKS)}
    for t in range(len(side_labels_over_time) - 1):
        src_labels = side_labels_over_time[t]
        dst_labels = side_labels_over_time[t + 1]
        n = min(len(src_labels), len(dst_labels))
        for i in range(n):
            src = src_labels[i]
            dst = dst_labels[i]
            if src in side_index and dst in side_index:
                total_counts[t] += 1.0
                if src != dst:
                    changed_counts[t] += 1.0


def write_transition_summary_csv(
    out_path: Path,
    transitions_by_persona_by_bot: dict[str, dict[str, np.ndarray]],
    final_transitions_by_persona_by_bot: dict[str, dict[str, np.ndarray]],
    changed_counts_by_persona_by_bot: dict[str, dict[str, np.ndarray]],
    total_counts_by_persona_by_bot: dict[str, dict[str, np.ndarray]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["persona_set", "bot", "metric", "value"]
    rows: list[dict[str, object]] = []
    for persona_set in sorted(transitions_by_persona_by_bot.keys()):
        trans_by_bot = transitions_by_persona_by_bot[persona_set]
        final_trans_by_bot = final_transitions_by_persona_by_bot[persona_set]
        changed_by_bot = changed_counts_by_persona_by_bot[persona_set]
        total_by_bot = total_counts_by_persona_by_bot[persona_set]
        for bot in ("off", "on"):
            step_mat = trans_by_bot.get(bot, np.zeros((len(PERSONA_BLOCKS), len(PERSONA_BLOCKS))))
            final_mat = final_trans_by_bot.get(bot, np.zeros((len(PERSONA_BLOCKS), len(PERSONA_BLOCKS))))
            total_step = float(np.sum(step_mat))
            total_final = float(np.sum(final_mat))
            step_stay = float(np.trace(step_mat) / total_step) if total_step > 0 else 0.0
            final_stay = float(np.trace(final_mat) / total_final) if total_final > 0 else 0.0
            changed = changed_by_bot.get(bot, np.zeros(0, dtype=float))
            totals = total_by_bot.get(bot, np.zeros(0, dtype=float))
            with np.errstate(divide="ignore", invalid="ignore"):
                rates = np.divide(changed, totals, where=totals > 0)
            mean_rate = float(np.mean(rates)) if len(rates) else 0.0
            peak_rate = float(np.max(rates)) if len(rates) else 0.0
            peak_step = int(np.argmax(rates) + 1) if len(rates) else 0
            rows.extend(
                [
                    {"persona_set": persona_set, "bot": bot, "metric": "step_stay_rate", "value": round(step_stay, 6)},
                    {"persona_set": persona_set, "bot": bot, "metric": "initial_to_final_stay_rate", "value": round(final_stay, 6)},
                    {"persona_set": persona_set, "bot": bot, "metric": "mean_step_change_rate", "value": round(mean_rate, 6)},
                    {"persona_set": persona_set, "bot": bot, "metric": "peak_step_change_rate", "value": round(peak_rate, 6)},
                    {"persona_set": persona_set, "bot": bot, "metric": "peak_step_change_rate_step", "value": peak_step},
                ]
            )

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Vote summary CSV
# ---------------------------------------------------------------------------

def write_vote_summary_csv(
    out_path: Path,
    matrix_rows: list[dict[str, object]],
    final_t: int,
) -> None:
    """Write a concise CSV of initial vs final votes by condition (semantic only)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "graph", "persona_set", "bot", "seed", "phase",
        "support", "against", "abstain",
    ]
    csv_rows: list[dict[str, object]] = []
    for row in matrix_rows:
        if str(row.get("model", "")) != "semantic":
            continue
        t = int(row["t"])
        if t not in (0, final_t):
            continue
        if row.get("vote_support_count") is None:
            continue
        csv_rows.append({
            "graph": row["graph"],
            "persona_set": row["persona_set"],
            "bot": row["bot"],
            "seed": row["seed"],
            "phase": "initial" if t == 0 else "final",
            "support": int(float(row["vote_support_count"])),
            "against": int(float(row["vote_against_count"])),
            "abstain": int(float(row["vote_abstain_count"])),
        })

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)


# ---------------------------------------------------------------------------
# Graph structure summary CSV
# ---------------------------------------------------------------------------

GRAPH_SUMMARY_FIELDS = [
    "graph", "persona_set", "bot", "seed",
    "nodes", "edges", "density", "components", "isolates",
    "avg_degree", "min_degree", "max_degree",
    "local_edges", "long_range_edges",
    "avg_clustering", "transitivity",
    "avg_shortest_path", "diameter",
    "degree_assortativity", "modularity",
    "bridge_edge_fraction",
    "avg_betweenness", "avg_eigenvector",
    "bot_degree",
]


def write_graph_structure_summary_csv(
    out_path: Path,
    matrix_rows: list[dict[str, object]],
) -> None:
    """Write one row per (graph, persona_set, bot, seed) at t=0 with graph metrics."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple] = set()
    csv_rows: list[dict[str, object]] = []

    for row in matrix_rows:
        if int(row["t"]) != 0:
            continue
        if str(row.get("model", "")) != "semantic":
            continue
        key = (str(row["graph"]), str(row["persona_set"]), str(row["bot"]), int(row["seed"]))
        if key in seen:
            continue
        seen.add(key)
        csv_rows.append({
            "graph": row["graph"],
            "persona_set": row["persona_set"],
            "bot": row["bot"],
            "seed": row["seed"],
            "nodes": row.get("graph_nodes"),
            "edges": row.get("graph_edges"),
            "density": row.get("graph_density"),
            "components": row.get("graph_components"),
            "isolates": row.get("graph_isolates"),
            "avg_degree": row.get("graph_avg_degree"),
            "min_degree": row.get("graph_min_degree"),
            "max_degree": row.get("graph_max_degree"),
            "local_edges": row.get("graph_local_edges"),
            "long_range_edges": row.get("graph_long_range_edges"),
            "avg_clustering": row.get("graph_avg_clustering"),
            "transitivity": row.get("graph_transitivity"),
            "avg_shortest_path": row.get("graph_avg_shortest_path"),
            "diameter": row.get("graph_diameter"),
            "degree_assortativity": row.get("graph_degree_assortativity"),
            "modularity": row.get("graph_modularity"),
            "bridge_edge_fraction": row.get("graph_bridge_edge_fraction"),
            "avg_betweenness": row.get("graph_avg_betweenness"),
            "avg_eigenvector": row.get("graph_avg_eigenvector"),
            "bot_degree": row.get("bot_degree"),
        })

    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=GRAPH_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)

