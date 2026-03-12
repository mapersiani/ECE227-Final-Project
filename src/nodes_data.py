"""
Helpers for loading external node/persona data.
"""

import json
from pathlib import Path


def load_nodes_data(path: Path) -> list[dict]:
    """
    Load and normalize node definitions.

    Supported input formats:
      1) JSON list of node records with fields like {name, prompt, initial}
      2) JSON object with key "nodes" (e.g., senate_nodes.json), where each record
         may use fields like {name, communication_style, seed_opinion}

    Returned normalized records always include:
      - name
      - prompt
      - initial
      - party (optional source field, empty if unavailable)
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        normalized = []
        for rec in data:
            normalized.append(
                {
                    "name": str(rec.get("name", "")),
                    "prompt": str(rec.get("prompt", rec.get("style", "You are a neutral participant."))),
                    "initial": str(rec.get("initial", rec.get("seed_opinion", ""))),
                    "party": str(rec.get("party", "")),
                }
            )
        return normalized

    if isinstance(data, dict) and isinstance(data.get("nodes"), list):
        normalized = []
        for rec in data["nodes"]:
            prompt = rec.get("prompt")
            if not prompt:
                style = str(rec.get("communication_style", "")).strip()
                background = str(rec.get("background", "")).strip()
                prompt = " ".join([x for x in [style, background] if x]).strip()
            if not prompt:
                prompt = "You are a neutral participant."
            normalized.append(
                {
                    "name": str(rec.get("name", "")),
                    "prompt": str(prompt),
                    "initial": str(rec.get("seed_opinion", rec.get("initial", ""))),
                    "party": str(rec.get("party", "")),
                }
            )
        return normalized

    raise ValueError("unsupported nodes data format")
