"""
Evolutionary game simulation on a network with node-specific personas.

Strategies are discrete camps:
0=left, 1=center_left, 2=center_right, 3=right.
Update rule uses Fermi imitation with persona-adjusted switching resistance.
"""

from typing import Optional

import networkx as nx
import numpy as np

from src.config import EGT_NODE_PERSONAS, EGT_STRATEGIES

_STRATEGY_INDEX = {name: i for i, name in enumerate(EGT_STRATEGIES)}

# Distance-based payoff matrix:
# same camp gets highest payoff, adjacent camps mildly positive, opposite camps negative.
PAYOFF_MATRIX = np.array(
    [
        [1.0, 0.4, -0.2, -0.8],
        [0.4, 1.0, 0.4, -0.2],
        [-0.2, 0.4, 1.0, 0.4],
        [-0.8, -0.2, 0.4, 1.0],
    ],
    dtype=float,
)


def _persona_by_node(n: int) -> list[dict]:
    """
    Return persona configs indexed by node id [0..n-1].
    Falls back to neutral persona if a node is missing.
    """
    defaults = [
        {
            "node_id": i,
            "block": "center_left",
            "openness": 0.5,
            "stubbornness": 0.5,
            "prompt": "Neutral policy participant with moderate views.",
        }
        for i in range(n)
    ]
    for p in EGT_NODE_PERSONAS:
        idx = int(p["node_id"])
        if 0 <= idx < n:
            defaults[idx] = p
    return defaults


def _infer_group_from_name(name: str) -> str:
    """Infer camp label from node name prefix."""
    if name.startswith("center_left_"):
        return "center_left"
    if name.startswith("center_right_"):
        return "center_right"
    if name.startswith("left_"):
        return "left"
    if name.startswith("right_"):
        return "right"
    return "center_left"


def _personas_from_nodes_data(nodes_data: list[dict]) -> list[dict]:
    """
    Build EGT personas from external nodes.json-style records.
    """
    group_defaults = {
        "left": {"openness": 0.40, "stubbornness": 0.68},
        "center_left": {"openness": 0.58, "stubbornness": 0.41},
        "center_right": {"openness": 0.54, "stubbornness": 0.46},
        "right": {"openness": 0.36, "stubbornness": 0.72},
    }
    personas: list[dict] = []
    for i, rec in enumerate(nodes_data):
        name = str(rec.get("name", ""))
        block = _infer_group_from_name(name)
        defaults = group_defaults.get(block, group_defaults["center_left"])
        personas.append(
            {
                "node_id": i,
                "name": name or f"node_{i}",
                "block": block,
                "openness": float(rec.get("openness", defaults["openness"])),
                "stubbornness": float(rec.get("stubbornness", defaults["stubbornness"])),
                "prompt": str(rec.get("prompt", "Neutral policy participant.")),
                "initial": str(rec.get("initial", "")),
            }
        )
    return personas


def _initial_strategies_from_personas(personas: list[dict]) -> np.ndarray:
    """Initialize each node strategy from persona block label."""
    s = np.zeros(len(personas), dtype=int)
    for i, p in enumerate(personas):
        s[i] = _STRATEGY_INDEX.get(p["block"], 1)
    return s


def _compute_payoffs(G: nx.Graph, strategies: np.ndarray) -> np.ndarray:
    """Sum pairwise payoffs against all neighbors."""
    n = G.number_of_nodes()
    payoffs = np.zeros(n, dtype=float)
    for i in range(n):
        si = strategies[i]
        for j in G.neighbors(i):
            payoffs[i] += PAYOFF_MATRIX[si, strategies[j]]
    return payoffs


def _fermi_prob(delta: float, beta: float) -> float:
    """Fermi imitation probability."""
    return float(1.0 / (1.0 + np.exp(-beta * delta)))


def run_egt(
    G: nx.Graph,
    steps: int = 20,
    beta: float = 2.5,
    switch_cost: float = 0.2,
    seed: Optional[int] = None,
    nodes_data: Optional[list[dict]] = None,
    topic: Optional[str] = None,
) -> dict:
    """
    Run asynchronous EGT dynamics with persona-aware resistance to switching.

    Returns dict with:
      - strategies_history: list[np.ndarray], t=0..steps
      - share_history: list[np.ndarray], share of 4 camps per step
      - polarization_history: list[float], variance of numeric strategy index
      - personas: list[dict], per-node persona definitions
    """
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()
    if nodes_data is not None:
        if len(nodes_data) != n:
            raise ValueError(f"nodes_data has {len(nodes_data)} entries but graph has {n} nodes")
        personas = _personas_from_nodes_data(nodes_data)
    else:
        personas = _persona_by_node(n)
    strategies = _initial_strategies_from_personas(personas)

    strategies_history = [strategies.copy()]
    share_history = [np.bincount(strategies, minlength=4) / n]
    polarization_history = [float(np.var(strategies.astype(float)))]

    for _ in range(steps):
        payoffs = _compute_payoffs(G, strategies)
        order = rng.permutation(n)
        for i in order:
            neighbors = list(G.neighbors(i))
            if not neighbors:
                continue
            j = int(rng.choice(neighbors))
            if i == j:
                continue

            # Persona modifies effective update pressure:
            # openness boosts imitation, stubbornness and switch cost resist it.
            openness = float(personas[i]["openness"])
            stubbornness = float(personas[i]["stubbornness"])
            same_strategy = strategies[i] == strategies[j]
            resistance = 0.0 if same_strategy else switch_cost * (1.0 + stubbornness)
            delta = (payoffs[j] - payoffs[i]) * (0.5 + openness) - resistance
            p = _fermi_prob(delta, beta)
            if rng.random() < p:
                strategies[i] = strategies[j]

        strategies_history.append(strategies.copy())
        share_history.append(np.bincount(strategies, minlength=4) / n)
        polarization_history.append(float(np.var(strategies.astype(float))))

    return {
        "strategies_history": strategies_history,
        "share_history": share_history,
        "polarization_history": polarization_history,
        "personas": personas,
        "topic": topic,
    }
