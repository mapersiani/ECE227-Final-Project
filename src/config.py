"""Configuration and constants for the simulation."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Network settings
DEFAULT_N = 20  # SBM: 4 blocks × 5 nodes

# Simulation
DEFAULT_TOPIC = "AI Regulation"
DEFAULT_STEPS = 5

# Persona templates aligned with SBM blocks: left, center_left, center_right, right
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
