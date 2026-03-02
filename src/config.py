"""
Configuration and constants for the simulation.

Loads .env for Ollama and HF settings. Defines persona prompts (aligned with SBM blocks:
left, center_left, center_right, right) and simulation defaults.
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# Ollama (local LLM). No API key required.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Network: SBM has 20 nodes, 4 blocks × 5 nodes each
DEFAULT_N = 20

# Simulation defaults
DEFAULT_TOPIC = "Government Environmental Regulation"
DEFAULT_STEPS = 5

#RGG parameters
RGG_RADIUS= float(os.getenv("RGG_RADIUS", "0.3"))  # distance threshold for local edges
LONG_RANGE_PROB= float(os.getenv("LONG_RANGE_PROB", "0.3"))  # fraction of nodes with long-range ties
LONG_RANGE_K= int(os.getenv("LONG_RANGE_K", "2"))  # number of long-range ties per selected node

def _load_nodes_json() -> list:
    """Parse nodes.json (may contain smart quotes or minor formatting issues)."""
    nodes_path = Path(__file__).parent.parent / "nodes.json"
    with open(nodes_path, encoding="utf-8") as f:
        content = f.read()

    raw_blocks = re.findall(r'\{[^{}]+\}', content, re.DOTALL)
    nodes = []
    for block in raw_blocks:
        name = re.search(r'"name":\s*"([^"]+)"', block)
        prompt = re.search(r'"prompt":\s*"(.*?)"(?=,\s*")', block, re.DOTALL)
        style = re.search(r'"style":\s*"(.*?)"(?=,\s*"|\s*\})', block, re.DOTALL)
        initial = re.search(r'"initial":\s*"(.*?)"(?=\s*\})', block, re.DOTALL)
        if name and prompt and initial:
            full_prompt = prompt.group(1).strip()
            if style:
                full_prompt += f"\n\nCommunication style: {style.group(1).strip()}"
            nodes.append({
                "name": name.group(1),
                "prompt": full_prompt,
                "initial": initial.group(1).strip(),
            })
    return nodes


# Load all 34 personas from nodes.json at import time
NODES = _load_nodes_json()

# Node names in order (used to build the graph)
NODE_NAMES = [n["name"] for n in NODES]

# Quick lookup: name → node dict
NODE_BY_NAME = {n["name"]: n for n in NODES}

