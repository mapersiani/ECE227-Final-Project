"""
Configuration and constants for the simulation.

Defines canonical, hard-coded experiment settings to avoid configuration drift.
Persona content is sourced from data/nodes.json or data/senate_nodes.json at runtime.
"""

# Ollama runtime (kept fixed for reproducibility).
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"

# Experiment constants (single source of truth)
TOPIC = "H.R.6938 - Appropriations Act, 2026. This bill cuts the EPA's budget by 4%, restricts the EPA from enforcing the 2015 Ozone Air Quality Standards, and expedites offshore oil and gas production to support an 'energy dominance agenda'."
# Canonical 3-seed set for reproducible multi-run comparisons.
SEED_LIST = [11]
DEFAULT_SEED = SEED_LIST[0]
SIMULATION_STEPS = 10
DEFAULT_ER_EDGE_PROB = 0.10
BOT_DEPLOY_STEPS = (1, 5, 10)
# Logging policy: compact per-step summaries only.
DEFAULT_LOG_MODE = "summary"
# Prompt budget controls for semantic updates (speed/stability).
MAX_NEIGHBORS_PER_UPDATE = 12
MAX_CHARS_PER_NEIGHBOR = 600
# Persona blocks: party_firstname_lastname naming
PERSONA_BLOCKS = ("democrat", "republican", "independent")
PERSONA_BLOCK_LAYOUT = {
    "democrat": (0, 0.20),
    "republican": (1, 0.80),
    "independent": (2, 0.50),
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
RGG_RADIUS = 0.20 # radius of the RGG graph for local ties
LONG_RANGE_FRACTION = 0.30 # fraction of long range neighbors
LONG_RANGE_K = 2 # max number of long range connection per node
