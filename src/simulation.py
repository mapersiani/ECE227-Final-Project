"""
Discrete-time simulation engine for semantic opinion dynamics on an RGG.

Each step:
  1. Opinion update: each agent reads neighbors' current opinions → LLM → new opinion
  2. Persona drift (every `persona_drift_every` steps): each agent reads neighbors'
     worldviews → LLM → subtly updated persona prompt used for future opinion calls

Tracks opinion variance, persona drift, and full graph metrics over time.
"""

from typing import Callable, List, Optional

import networkx as nx
import numpy as np

from src.agent import Agent
from src.config import DEFAULT_TOPIC, NODES, NODE_NAMES
from src.llm_client import get_updated_opinion, get_updated_persona
from src.measurement import embed_opinions, semantic_variance, compute_graph_metrics, SimulationRecord


def create_agents(
    G: nx.Graph,
    topic: str = DEFAULT_TOPIC,
    seed: Optional[int] = None,
) -> List[Agent]:
    """
    Create one Agent per node using the full persona and seed opinion from nodes.json.

    Each agent starts with:
      - persona_prompt: the nodes.json character description (will drift over simulation)
      - initial_opinion: the nodes.json seed opinion
    """
    agents = []
    n = G.number_of_nodes()
    node_names_in_graph = nx.get_node_attributes(G, "name")

    for i in range(n):
        name = node_names_in_graph.get(i, NODE_NAMES[i] if i < len(NODE_NAMES) else "unknown")
        node_data = next((nd for nd in NODES if nd["name"] == name), None)
        if node_data is None:
            persona_prompt = f"You are a thoughtful citizen with views on {topic}."
            initial_opinion = f"I have mixed feelings about {topic}."
        else:
            persona_prompt = node_data["prompt"]
            initial_opinion = node_data["initial"]

        agents.append(Agent(
            node_id=i,
            persona_prompt=persona_prompt,
            initial_opinion=initial_opinion,
        ))
    return agents


def step_semantic(
    G: nx.Graph,
    agents: List[Agent],
    topic: str,
) -> None:
    """
    Opinion update step: snapshot all opinions, then each agent calls LLM with
    neighbors' opinions (using agent's CURRENT persona, which may have drifted).
    Updates agents in place.
    """
    opinion_snapshot = [a.current_opinion for a in agents]
    for i in range(G.number_of_nodes()):
        if agents[i].is_bot:
            continue
        neighbors = list(G.neighbors(i))
        neighbor_opinions = [opinion_snapshot[j] for j in neighbors]
        if not neighbor_opinions:
            continue
        new_opinion = get_updated_opinion(
            persona=agents[i].persona_prompt,   # uses current (possibly drifted) persona
            topic=topic,
            neighbor_opinions=neighbor_opinions,
        )
        agents[i].update_opinion(new_opinion)


def step_persona_drift(
    G: nx.Graph,
    agents: List[Agent],
    topic: str,
) -> None:
    """
    Persona drift step: each agent reads neighbors' CURRENT personas and opinions,
    then calls LLM to produce a slightly shifted persona description.

    The agent's ORIGINAL nodes.json persona is always passed as an immutable anchor
    so the LLM cannot drift the character beyond recognition across multiple steps.
    Drift is a nuance/emphasis shift from that fixed baseline, not accumulated
    drift stacked on top of previous drift.

    Called less frequently than opinion updates (every N steps) because
    real identity shift is slower than stated opinion change.
    """
    persona_snapshot = [a.persona_prompt for a in agents]
    opinion_snapshot = [a.current_opinion for a in agents]

    for i in range(G.number_of_nodes()):
        if agents[i].is_bot:
            continue
        neighbors = list(G.neighbors(i))
        if not neighbors:
            continue
        neighbor_personas = [persona_snapshot[j] for j in neighbors]
        neighbor_opinions = [opinion_snapshot[j] for j in neighbors]

        new_persona = get_updated_persona(
            original_persona=agents[i].initial_persona,   # frozen nodes.json anchor
            current_persona=agents[i].persona_prompt,     # current drifted state (context only)
            neighbor_personas=neighbor_personas,
            neighbor_opinions=neighbor_opinions,
            topic=topic,
        )
        agents[i].update_persona(new_persona)


def run_semantic(
    G: nx.Graph,
    agents: List[Agent],
    topic: str,
    steps: int = 5,
    persona_drift_every: int = 2,
    compute_metrics: bool = True,
    on_step: Optional[Callable[[int, List[Agent], Optional[dict]], None]] = None,
    show_progress: bool = True,
) -> SimulationRecord:
    """
    Run the full semantic simulation with opinion updates and persona drift.

    Args:
        G: The RGG network
        agents: Agent list from create_agents()
        topic: Discussion topic
        steps: Number of simulation steps
        persona_drift_every: Run persona drift every N opinion steps (default 2).
                             Set to 0 to disable persona drift entirely.
        compute_metrics: Whether to compute full graph+semantic metrics each step
                         (slower but enables analysis). Set False for quick runs.
        on_step: Optional callback(t, agents, metrics_or_None) after each step
        show_progress: Show tqdm progress bar

    Returns:
        SimulationRecord with all step snapshots and node metrics.
    """
    from tqdm import tqdm

    record = SimulationRecord(
        topic=topic,
        steps=steps,
        graph_params={
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "local_edges": sum(1 for _, _, d in G.edges(data=True) if d.get("edge_type") == "local"),
            "long_range_edges": sum(1 for _, _, d in G.edges(data=True) if d.get("edge_type") == "long_range"),
        },
    )
    full_metrics_by_step = []  # store complete metrics (incl. node_metrics) per step

    # ── t=0 baseline ──
    if compute_metrics:
        m0 = compute_graph_metrics(G, agents)
        record.add_step(0, m0)
        full_metrics_by_step.append(m0)
        print(f"\nt=0  opinion_var={m0['opinion_variance']:.4f}  "
              f"persona_drift={m0['persona_drift_mean']:.4f}  "
              f"polarization={m0['opinion_polarization']:.3f}")
    else:
        opinions = [a.current_opinion for a in agents]
        v = semantic_variance(embed_opinions(opinions))
        record.add_step(0, {"opinion_variance": v})

    step_range = range(1, steps + 1)
    if show_progress:
        step_range = tqdm(step_range, desc="Simulation", unit="step")

    for t in step_range:
        # 1. Opinion update (every step)
        step_semantic(G, agents, topic)

        # 2. Persona drift (every N steps)
        if persona_drift_every > 0 and t % persona_drift_every == 0:
            if show_progress:
                tqdm.write(f"  [t={t}] Running persona drift update...")
            step_persona_drift(G, agents, topic)

        # 3. Metrics
        if compute_metrics:
            m = compute_graph_metrics(G, agents)
            record.add_step(t, m)
            full_metrics_by_step.append(m)
            if show_progress:
                tqdm.write(
                    f"  t={t}  opinion_var={m['opinion_variance']:.4f}  "
                    f"persona_drift={m['persona_drift_mean']:.4f}  "
                    f"polarization={m['opinion_polarization']:.3f}"
                )
        else:
            opinions = [a.current_opinion for a in agents]
            v = semantic_variance(embed_opinions(opinions))
            record.add_step(t, {"opinion_variance": v})

        if on_step:
            on_step(t, agents, full_metrics_by_step[-1] if compute_metrics else None)

    # Attach full per-step node metrics for downstream analysis
    record._full_metrics = full_metrics_by_step
    return record