"""
Configuration and constants for the simulation.

Loads .env for Ollama and HF settings. Defines persona prompts (aligned with SBM blocks:
left, center_left, center_right, right) and simulation defaults.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# Ollama (local LLM). No API key required.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

# Network: SBM has 20 nodes, 4 blocks × 5 nodes each
DEFAULT_N = 20

# Simulation defaults (topic fixed for this study)
DEFAULT_TOPIC = "Government Environmental Regulations"
DEFAULT_STEPS = 5

# Persona prompts for each block. Used to initialize agents and steer LLM responses.
PERSONAS = [
    {
        "name": "left",
        "prompt": "You hold left-leaning views. You favor strong regulation, collective action, and skepticism of corporate power.",
    },
    {
        "name": "center_left",
        "prompt": "You are center-left. You support regulation with room for innovation, and balance market and social concerns.",
    },
    {
        "name": "center_right",
        "prompt": "You are center-right. You prefer limited regulation, trusting markets while acknowledging some need for guardrails.",
    },
    {
        "name": "right",
        "prompt": "You hold right-leaning views. You favor minimal regulation and believe free markets and voluntary action are best.",
    },
]
