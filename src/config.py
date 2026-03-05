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

DEFAULT_N = 36

# Experiment constants (single source of truth)
TOPIC = "Government Environmental Regulations"
SEED_LIST = [11, 23, 42, 57, 89]
DEFAULT_SEED = SEED_LIST[2]
SIMULATION_STEPS = 10
BOT_INJECTION_STEP = 0
PERSONA_BLOCKS = ("left", "center_left", "center_right", "right")
PERSONA_BLOCK_LAYOUT = {
    "left": (0, 0.10),
    "center_left": (1, 0.35),
    "center_right": (2, 0.65),
    "right": (3, 0.90),
}
DEGROOT_SCALAR_BY_BLOCK = {
    "left": 0.0,
    "center_left": 1.0 / 3.0,
    "center_right": 2.0 / 3.0,
    "right": 1.0,
}

# Backward-compatible aliases used across existing modules
DEFAULT_TOPIC = TOPIC
DEFAULT_STEPS = SIMULATION_STEPS

# RGG + long-range defaults
RGG_RADIUS = float(os.getenv("RGG_RADIUS", "0.30"))
LONG_RANGE_FRACTION = float(os.getenv("LONG_RANGE_FRACTION", "0.30"))
LONG_RANGE_K = int(os.getenv("LONG_RANGE_K", "2"))

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
