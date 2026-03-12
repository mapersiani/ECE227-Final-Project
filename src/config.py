"""
Configuration and constants for the simulation.

Defines canonical, hard-coded experiment settings to avoid configuration drift.
Persona content is sourced from data files (e.g., nodes.json/senate_nodes.json) at runtime.
"""

# Ollama runtime (kept fixed for reproducibility).
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"

# Experiment constants (single source of truth)
TOPIC = "Government Environmental Regulations"
# Canonical seed set for reproducible runs.
SEED_LIST = [42]
DEFAULT_SEED = SEED_LIST[-1]
SIMULATION_STEPS = 5
DEFAULT_ER_EDGE_PROB = 0.15
DEFAULT_BOT_POST_PROB = 0.80  # Probability that a bot posts in a given step.
BOT_INJECTION_STEP = 0
BOT_DEPLOY_STEPS = (1, 5, 10)
# Logging policy: compact per-step summaries only.
DEFAULT_LOG_MODE = "summary"
# Prompt budget controls for semantic updates (speed/stability).
MAX_NEIGHBORS_PER_UPDATE = 6
MAX_CHARS_PER_NEIGHBOR = 320
# Persona blocks: party_firstname_lastname naming
PERSONA_BLOCKS = ("democrat", "republican")
PERSONA_BLOCK_LAYOUT = {
    "democrat": (0, 0.20),
    "republican": (1, 0.80),
}


def side_from_name(name: str) -> str:
    """Derive block from node name (party_firstname_lastname)."""
    for block in PERSONA_BLOCKS:
        if name.startswith(block):
            return block
    return "unknown"


# Aliases used across modules
DEFAULT_TOPIC = TOPIC
DEFAULT_STEPS = SIMULATION_STEPS

# RGG + long-range defaults
RGG_RADIUS = 0.30  # radius of the RGG graph for local ties
LONG_RANGE_FRACTION = 0.30  # fraction of long-range neighbors
LONG_RANGE_K = 2  # max number of long-range connections per node