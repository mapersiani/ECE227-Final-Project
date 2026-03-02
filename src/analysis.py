"""
Post-simulation analysis and visualization.

Given a SimulationRecord and the agents/graph after a run, produces:
  - Time-series plots: opinion variance, persona drift, polarization
  - Node-level scatter: persona drift vs centrality, drift vs degree
  - Network graph colored by persona drift magnitude
  - Block-level opinion divergence heatmap
  - Printed summary table of all scalar metrics
  - Optional: save all data to JSON for external analysis
"""

import json
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np

from src.measurement import SimulationRecord, embed_opinions, pairwise_cosine_distances

BLOCK_COLORS = {0: "#3B82F6", 1: "#22C55E", 2: "#F97316", 3: "#EF4444"}
BLOCK_LABELS = {0: "left", 1: "center_left", 2: "center_right", 3: "right"}


# ─── Time-series panel ─────────────────────────────────────────────────────────

def plot_timeseries(record: SimulationRecord, save_path: Optional[Path] = None) -> None:
    """3-panel time-series: opinion variance, persona drift, and polarization."""
    snapshots = record.step_snapshots
    ts = [s["t"] for s in snapshots]

    keys = ["opinion_variance", "persona_drift_mean", "opinion_polarization"]
    labels = ["Opinion Variance\n(semantic spread)", "Persona Drift\n(mean cosine Δ from t=0)", "Polarization\n(between/within block ratio)"]
    colors = ["steelblue", "darkorchid", "tomato"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, label, color in zip(axes, keys, labels, colors):
        vals = [s.get(key, float("nan")) for s in snapshots]
        ax.plot(ts, vals, marker="o", color=color, linewidth=2)
        ax.fill_between(ts, vals, alpha=0.12, color=color)
        ax.set_xlabel("Timestep")
        ax.set_ylabel(label)
        ax.set_title(label.split("\n")[0])
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Opinion Dynamics — {record.topic}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")
    plt.close(fig)


# ─── Node-level scatter ────────────────────────────────────────────────────────

def plot_node_scatter(full_metrics: List[dict], save_path: Optional[Path] = None) -> None:
    """
    Scatter plots at final timestep:
      Left:  betweenness centrality vs persona drift
      Right: degree vs persona drift
    Points colored by block.
    """
    if not full_metrics:
        return
    final_nodes = full_metrics[-1]["node_metrics"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, x_key, x_label in [
        (axes[0], "betweenness", "Betweenness Centrality"),
        (axes[1], "degree",      "Node Degree"),
    ]:
        for block in range(4):
            subset = [n for n in final_nodes if n["block"] == block]
            if not subset:
                continue
            xs = [n[x_key] for n in subset]
            ys = [n["persona_drift"] for n in subset]
            ax.scatter(xs, ys, c=BLOCK_COLORS[block], label=BLOCK_LABELS[block],
                       s=80, alpha=0.8, edgecolors="white", linewidths=0.5)

        # Trend line
        all_x = np.array([n[x_key] for n in final_nodes])
        all_y = np.array([n["persona_drift"] for n in final_nodes])
        if np.std(all_x) > 1e-9:
            m, b = np.polyfit(all_x, all_y, 1)
            xs_line = np.linspace(all_x.min(), all_x.max(), 100)
            ax.plot(xs_line, m * xs_line + b, "k--", alpha=0.4, linewidth=1, label="trend")

        ax.set_xlabel(x_label)
        ax.set_ylabel("Persona Drift (cosine Δ from initial)")
        ax.set_title(f"Persona Drift vs {x_label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Node-Level Persona Drift Analysis (Final Timestep)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")
    plt.close(fig)


# ─── Network colored by drift ──────────────────────────────────────────────────

def plot_drift_network(
    G: nx.Graph,
    full_metrics: List[dict],
    save_path: Optional[Path] = None,
) -> None:
    """Draw the RGG with nodes colored by persona drift magnitude (cool → hot colormap)."""
    if not full_metrics:
        return
    final_nodes = {n["node_id"]: n for n in full_metrics[-1]["node_metrics"]}

    pos = {i: G.nodes[i]["pos"] for i in G.nodes()}
    drift_vals = np.array([final_nodes.get(i, {}).get("persona_drift", 0.0) for i in G.nodes()])

    cmap = plt.cm.plasma
    norm = mcolors.Normalize(vmin=drift_vals.min(), vmax=drift_vals.max())
    node_colors = [cmap(norm(d)) for d in drift_vals]

    local_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "local"]
    long_edges  = [(u, v) for u, v, d in G.edges(data=True) if d.get("edge_type") == "long_range"]

    fig, ax = plt.subplots(figsize=(12, 8))
    nx.draw_networkx_edges(G, pos, edgelist=local_edges, alpha=0.25, edge_color="#aaa", ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=long_edges, alpha=0.55, edge_color="#A855F7",
                           style="dashed", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=220, ax=ax)

    # Labels for high-drift nodes
    drift_threshold = np.percentile(drift_vals, 75)
    labels = {
        i: G.nodes[i].get("name", str(i)).split("_")[-1]
        for i in G.nodes()
        if drift_vals[i] >= drift_threshold
    }
    nx.draw_networkx_labels(G, pos, labels, font_size=6, ax=ax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Persona Drift (cosine Δ from initial)", shrink=0.7)

    ax.set_title("Persona Drift by Node — RGG Network", fontsize=13, fontweight="bold")
    ax.set_xlabel("Ideological position (left → right)")
    ax.set_ylabel("Engagement level")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")
    plt.close(fig)


# ─── Block opinion heatmap ─────────────────────────────────────────────────────

def plot_block_opinion_heatmap(
    agents,
    G: nx.Graph,
    save_path: Optional[Path] = None,
) -> None:
    """
    Heatmap of mean pairwise cosine distance between opinions, grouped by block.
    Shows whether blocks have converged or remained polarized.
    """
    from src.measurement import _get_model
    real_agents = [a for a in agents if not a.is_bot]
    blocks = [G.nodes[a.node_id].get("block", 0) for a in real_agents]
    opinions = [a.current_opinion for a in real_agents]

    model = _get_model(show_progress=False)
    embs = model.encode(opinions, convert_to_numpy=True, show_progress_bar=False)
    dist = pairwise_cosine_distances(embs)

    # 4×4 block-averaged distance matrix
    n_blocks = 4
    block_dist = np.zeros((n_blocks, n_blocks))
    counts = np.zeros((n_blocks, n_blocks))
    for i, bi in enumerate(blocks):
        for j, bj in enumerate(blocks):
            if i != j:
                block_dist[bi, bj] += dist[i, j]
                counts[bi, bj] += 1
    with np.errstate(invalid="ignore"):
        block_dist = np.where(counts > 0, block_dist / counts, 0)

    labels = [BLOCK_LABELS[b] for b in range(n_blocks)]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(block_dist, cmap="RdYlGn_r", vmin=0, vmax=block_dist.max())
    plt.colorbar(im, ax=ax, label="Mean Cosine Distance")
    ax.set_xticks(range(n_blocks)); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(n_blocks)); ax.set_yticklabels(labels)
    ax.set_title("Block-Level Opinion Distance Heatmap", fontsize=12, fontweight="bold")
    ax.set_xlabel("Block"); ax.set_ylabel("Block")

    for i in range(n_blocks):
        for j in range(n_blocks):
            ax.text(j, i, f"{block_dist[i,j]:.3f}", ha="center", va="center",
                    fontsize=9, color="white" if block_dist[i, j] > block_dist.max() * 0.6 else "black")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")
    plt.close(fig)


# ─── Persona drift per block ───────────────────────────────────────────────────

def plot_persona_drift_by_block(
    full_metrics: List[dict],
    save_path: Optional[Path] = None,
) -> None:
    """Line plot of mean persona drift over time, split by ideological block."""
    if not full_metrics:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for block in range(4):
        drift_over_time = []
        for step_metrics in full_metrics:
            nodes = [n for n in step_metrics["node_metrics"] if n["block"] == block]
            if nodes:
                drift_over_time.append(np.mean([n["persona_drift"] for n in nodes]))
            else:
                drift_over_time.append(float("nan"))
        ts = list(range(len(drift_over_time)))
        ax.plot(ts, drift_over_time, marker="o", color=BLOCK_COLORS[block],
                label=BLOCK_LABELS[block], linewidth=2)

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean Persona Drift (cosine Δ from initial)")
    ax.set_title("Persona Drift Over Time — By Ideological Block", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")
    plt.close(fig)


# ─── Summary table ─────────────────────────────────────────────────────────────

def print_summary_table(record: SimulationRecord, full_metrics: List[dict]) -> None:
    """Print a formatted table of all scalar metrics at each timestep."""
    scalar_keys = [
        "opinion_variance", "opinion_polarization",
        "persona_drift_mean", "persona_drift_std",
        "avg_clustering", "modularity", "degree_assortativity",
        "avg_betweenness", "avg_shortest_path",
    ]
    header = f"{'t':>3}  " + "  ".join(f"{k[:12]:>12}" for k in scalar_keys)
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for s in record.step_snapshots:
        row = f"{s['t']:>3}  " + "  ".join(
            f"{s.get(k, float('nan')):>12.4f}" for k in scalar_keys
        )
        print(row)
    print("─" * len(header))

    if full_metrics:
        print("\nTop 5 nodes by persona drift (final step):")
        final_nodes = sorted(full_metrics[-1]["node_metrics"], key=lambda n: n["persona_drift"], reverse=True)
        for n in final_nodes[:5]:
            print(f"  {n['name']:<40} drift={n['persona_drift']:.4f}  "
                  f"degree={n['degree']}  betweenness={n['betweenness']:.4f}  "
                  f"long_range={n['is_long_range']}")


# ─── Save to JSON ──────────────────────────────────────────────────────────────

def save_results_json(
    record: SimulationRecord,
    full_metrics: List[dict],
    agents,
    path: Path,
) -> None:
    """
    Save complete simulation results to JSON for external analysis
    (pandas, R, custom plots, etc.).
    """
    def _safe(v):
        if isinstance(v, (np.float32, np.float64, float)):
            return float(v)
        if isinstance(v, (np.int32, np.int64, int)):
            return int(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    out = {
        "topic": record.topic,
        "steps": record.steps,
        "graph_params": record.graph_params,
        "step_snapshots": [
            {k: _safe(v) for k, v in s.items()} for s in record.step_snapshots
        ],
        "full_metrics": [
            {
                k: (_safe(v) if k != "node_metrics" else [
                    {nk: _safe(nv) for nk, nv in nm.items()} for nm in v
                ])
                for k, v in m.items()
            }
            for m in full_metrics
        ],
        "agent_histories": [
            {
                "node_id": a.node_id,
                "name": next((n["name"] for n in [{"name": str(a.node_id)}] if True), str(a.node_id)),
                "is_bot": a.is_bot,
                "persona_drift_count": a.persona_drift_count(),
                "opinion_history": a.opinion_history,
                "persona_history": a.persona_history,
            }
            for a in agents
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Saved results JSON → {path}")


# ─── One-call full analysis ────────────────────────────────────────────────────

def run_full_analysis(
    record: SimulationRecord,
    agents,
    G: nx.Graph,
    output_dir: Path,
) -> None:
    """
    Run all analysis plots and print summary. Call this after simulation completes.

    Saves:
        timeseries.png          — opinion variance, persona drift, polarization over time
        node_scatter.png        — drift vs centrality, drift vs degree
        drift_network.png       — RGG colored by drift magnitude
        block_heatmap.png       — block-vs-block opinion distance
        persona_drift_block.png — drift over time by ideological block
        results.json            — full numeric data for external analysis
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    full_metrics = getattr(record, "_full_metrics", [])

    print_summary_table(record, full_metrics)

    plot_timeseries(record,        save_path=output_dir / "timeseries.png")
    plot_node_scatter(full_metrics, save_path=output_dir / "node_scatter.png")
    plot_drift_network(G, full_metrics, save_path=output_dir / "drift_network.png")
    plot_block_opinion_heatmap(agents, G, save_path=output_dir / "block_heatmap.png")
    plot_persona_drift_by_block(full_metrics, save_path=output_dir / "persona_drift_block.png")
    save_results_json(record, full_metrics, agents, path=output_dir / "results.json")

    print(f"\nAll outputs saved to {output_dir}/")