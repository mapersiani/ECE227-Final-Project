"""
Configuration and constants for the simulation.

Defines canonical, hard-coded experiment settings to avoid configuration drift.
Persona content is sourced from data/nodes.json at runtime.
"""

# Ollama runtime (kept fixed for reproducibility).
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"

DEFAULT_N = 36

# Experiment constants (single source of truth)
TOPIC = "Government Environmental Regulations"
# Canonical 3-seed set for reproducible multi-run comparisons.
SEED_LIST = [11, 23, 42]
DEFAULT_SEED = SEED_LIST[2]
SIMULATION_STEPS = 10
DEFAULT_ER_EDGE_PROB = 0.15
DEFAULT_BOT_POST_PROB = 0.80 # Probability that a bot posts in a given step (vs. remaining silent).
BOT_INJECTION_STEP = 0
# Logging policy: compact per-step summaries only.
DEFAULT_LOG_MODE = "summary"
# Prompt budget controls for semantic updates (speed/stability).
MAX_NEIGHBORS_PER_UPDATE = 6
MAX_CHARS_PER_NEIGHBOR = 320
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
RGG_RADIUS = 0.30
LONG_RANGE_FRACTION = 0.30
LONG_RANGE_K = 2
