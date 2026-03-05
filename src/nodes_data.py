"""
Helpers for loading external node/persona data.
"""

import json
from pathlib import Path


def load_nodes_data(path: Path) -> list[dict]:
    """
    Load node definitions from JSON array.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("nodes data must be a JSON list")
    return data
