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

# Simulation defaults
DEFAULT_TOPIC = "AI Regulation"
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

# EGT defaults
DEFAULT_EGT_BETA = 2.5
DEFAULT_EGT_SWITCH_COST = 0.2

# 4-strategy order used by EGT simulations.
EGT_STRATEGIES = ("left", "center_left", "center_right", "right")

# Prototype texts used to map free-form opinions into 4 strategy camps via SBERT.
EGT_STRATEGY_PROTOTYPES = {
    "left": "Strong AI regulation is necessary to protect society from corporate abuse and systemic harms.",
    "center_left": "AI should be regulated with safeguards while still supporting responsible innovation.",
    "center_right": "AI policy should be light-touch, with targeted guardrails and market-led progress.",
    "right": "AI should face minimal regulation because free markets and voluntary standards work best.",
}

# 20 node-specific personas for EGT. Node id should match SBM node id.
# openness: willingness to imitate neighbors; stubbornness: resistance to switching.
EGT_NODE_PERSONAS = [
    {"node_id": 0, "block": "left", "openness": 0.35, "stubbornness": 0.70, "prompt": "Labor organizer focused on worker protections and strict AI oversight."},
    {"node_id": 1, "block": "left", "openness": 0.40, "stubbornness": 0.65, "prompt": "Public-interest lawyer emphasizing accountability and anti-monopoly safeguards."},
    {"node_id": 2, "block": "left", "openness": 0.30, "stubbornness": 0.75, "prompt": "Digital-rights activist skeptical of corporate self-regulation and opaque AI deployment."},
    {"node_id": 3, "block": "left", "openness": 0.45, "stubbornness": 0.60, "prompt": "Union-backed policy advocate pushing binding standards and enforcement."},
    {"node_id": 4, "block": "left", "openness": 0.38, "stubbornness": 0.68, "prompt": "Community organizer prioritizing equity, public safety, and democratic control of AI."},
    {"node_id": 5, "block": "center_left", "openness": 0.55, "stubbornness": 0.45, "prompt": "Pragmatic regulator supporting guardrails while preserving innovation incentives."},
    {"node_id": 6, "block": "center_left", "openness": 0.60, "stubbornness": 0.40, "prompt": "Policy analyst favoring evidence-based rules and periodic policy review."},
    {"node_id": 7, "block": "center_left", "openness": 0.58, "stubbornness": 0.42, "prompt": "Civic technologist balancing public benefit with startup ecosystem health."},
    {"node_id": 8, "block": "center_left", "openness": 0.62, "stubbornness": 0.35, "prompt": "Moderate lawmaker open to compromise, with focus on risk-tiered regulation."},
    {"node_id": 9, "block": "center_left", "openness": 0.57, "stubbornness": 0.43, "prompt": "Education-focused advocate supporting transparency and gradual implementation."},
    {"node_id": 10, "block": "center_right", "openness": 0.56, "stubbornness": 0.44, "prompt": "Market-oriented policy advisor preferring targeted, minimal intervention."},
    {"node_id": 11, "block": "center_right", "openness": 0.52, "stubbornness": 0.48, "prompt": "Business-friendly regulator supporting standards but avoiding heavy compliance burdens."},
    {"node_id": 12, "block": "center_right", "openness": 0.50, "stubbornness": 0.50, "prompt": "Innovation economist arguing for sandbox policies and selective oversight."},
    {"node_id": 13, "block": "center_right", "openness": 0.54, "stubbornness": 0.46, "prompt": "Industry liaison preferring agile governance over rigid rulemaking."},
    {"node_id": 14, "block": "center_right", "openness": 0.59, "stubbornness": 0.41, "prompt": "Tech policy moderate trusting competitive markets with clear safety triggers."},
    {"node_id": 15, "block": "right", "openness": 0.36, "stubbornness": 0.72, "prompt": "Libertarian commentator opposing broad regulation and favoring voluntary standards."},
    {"node_id": 16, "block": "right", "openness": 0.33, "stubbornness": 0.76, "prompt": "Free-market advocate warning that regulation will suppress entrepreneurship."},
    {"node_id": 17, "block": "right", "openness": 0.40, "stubbornness": 0.67, "prompt": "Small-government strategist emphasizing private governance and competition."},
    {"node_id": 18, "block": "right", "openness": 0.37, "stubbornness": 0.71, "prompt": "Pro-business analyst prioritizing growth over precautionary restrictions."},
    {"node_id": 19, "block": "right", "openness": 0.34, "stubbornness": 0.74, "prompt": "Civil-liberties conservative opposing centralized control over AI development."},
]
