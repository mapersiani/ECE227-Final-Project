"""
Node/persona loading for opinion dynamics simulations.

Loads persona data from JSON files. Node count is dynamic (determined from the file).
Supports multiple persona sets via persona_set flag (personas, senate).
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NODES_DIR = PROJECT_ROOT / "data"

# Map persona_set -> filename
PERSONA_SET_FILES = {
    "personas": "nodes.json",
    "senate": "senate_nodes.json",
}

# Cache per persona_set to avoid reloading
_CACHE: dict[str, list[dict]] = {}


def load_nodes(persona_set: str = "personas") -> list[dict]:
    """
    Load persona nodes from the configured JSON file.

    Args:
        persona_set: One of "personas" (data/nodes.json) or "senate" (data/senate_nodes.json)

    Returns:
        List of node dicts with keys: name, prompt, style, initial.
        Node names should follow party_firstname_lastname (e.g., democrat_Joe_Biden).
    """
    global _CACHE
    if persona_set not in PERSONA_SET_FILES:
        raise ValueError(
            f"Unknown persona_set '{persona_set}'. "
            f"Valid options: {list(PERSONA_SET_FILES.keys())}"
        )
    if persona_set in _CACHE:
        return _CACHE[persona_set]

    filename = PERSONA_SET_FILES[persona_set]
    path = NODES_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Nodes file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    # Handle wrapped format: {"nodes": [...], "metadata": {...}}
    if isinstance(raw, dict) and "nodes" in raw:
        nodes = raw["nodes"]
    elif isinstance(raw, list):
        nodes = raw
    else:
        raise ValueError(f"Unexpected JSON structure in {filename}: expected list or {{nodes: [...]}}")

    if not nodes:
        raise ValueError(f"Nodes file {filename} is empty")

    _CACHE[persona_set] = nodes
    return nodes


def node_count(persona_set: str = "personas") -> int:
    """Return the number of nodes for the given persona set."""
    return len(load_nodes(persona_set))
