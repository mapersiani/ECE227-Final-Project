"""
LLM client for opinion updates via Ollama.

Calls local Ollama API. No API key; requires Ollama running with a pulled model.
"""

import json
import urllib.error
import urllib.request
from typing import Sequence

from src.config import (
    MAX_CHARS_PER_NEIGHBOR,
    MAX_NEIGHBORS_PER_UPDATE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)


def _ollama_generate(prompt: str) -> str:
    """
    Send prompt to Ollama /api/generate. Returns model response text.
    Raises RuntimeError if Ollama is not running.
    """
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    body = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        if "Connection refused" in str(e) or "localhost" in str(e).lower():
            raise RuntimeError(
                "Ollama is not running. Start it: ollama serve (or ollama run llama3.2:3b)"
            ) from e
        raise
    return (data.get("response") or "").strip()


def _truncate_text(text: str, max_chars: int) -> str:
    cleaned = " ".join(str(text).split())
    if max_chars <= 0 or len(cleaned) <= max_chars:
        return cleaned
    if max_chars <= 3:
        return cleaned[:max_chars]
    return cleaned[: max_chars - 3] + "..."


def prepare_neighbor_opinions(
    neighbor_opinions: Sequence[str],
    max_neighbors: int = MAX_NEIGHBORS_PER_UPDATE,
    max_chars_per_neighbor: int = MAX_CHARS_PER_NEIGHBOR,
) -> list[str]:
    """
    Limit neighbor payload size to keep prompt cost stable and fast.
    """
    selected = list(neighbor_opinions)
    if max_neighbors > 0 and len(selected) > max_neighbors:
        selected = selected[:max_neighbors]

    prepared: list[str] = []
    for opinion in selected:
        trimmed = _truncate_text(opinion, max_chars=max_chars_per_neighbor).strip()
        if trimmed:
            prepared.append(trimmed)
    return prepared


def get_updated_opinion(
    persona: str,
    topic: str,
    neighbor_opinions: Sequence[str],
    memory: str = "",
    opinions_prepared: bool = False,
) -> str:
    """
    Ask Ollama for an updated opinion given persona, topic, and neighbor opinions.

    Args:
        persona: Agent's persona prompt
        topic: Discussion topic (e.g. "Government Environmental Regulations")
        neighbor_opinions: List of neighbor opinion texts
        memory: Optional prior context
        opinions_prepared: If True, skip neighbor preprocessing.

    Returns:
        New opinion text (1–2 sentences).
    """
    if opinions_prepared:
        prepared_neighbors = list(neighbor_opinions)
    else:
        prepared_neighbors = prepare_neighbor_opinions(neighbor_opinions)
    neighbor_text = "\n".join(f"- Neighbor {i + 1}: {o}" for i, o in enumerate(prepared_neighbors))
    memory_line = f"\nPrevious context or memory: {memory}\n" if memory else ""
    prompt = f"""You are simulating an agent in a social network opinion dynamics experiment.

{persona}

The topic under discussion is: {topic}

You have just read the following opinions from your neighbors:

{neighbor_text}
{memory_line}
In 1-2 concise sentences, state your updated opinion on this topic. Reflect how you are influenced by your neighbors' arguments (or lack thereof), but stay in character. Output only the opinion text, no meta-commentary."""
    return _ollama_generate(prompt)


def get_vote(
    persona: str,
    topic: str,
    opinion: str,
) -> str:
    """
    Ask the LLM to categorize its stance based on its persona and current opinion.
    Returns: 'SUPPORT', 'AGAINST', or 'ABSTAIN'
    """
    prompt = f"""You are simulating an agent in a social network opinion dynamics experiment.
    
{persona}

The topic under discussion is: {topic}

Your current opinion is: {opinion}

Based on your persona and current opinion, cast a vote on whether you SUPPORT or are AGAINST the topic.
If your persona generally opposes environmental regulations and favors fossil fuels, you should vote SUPPORT (since the bill cuts EPA funding and promotes oil/gas).
If your persona generally supports environmental regulations and opposes fossil fuels, you should vote AGAINST (since the bill cuts EPA funding and promotes oil/gas).

You must output exactly one of the following three words, and absolutely nothing else:
SUPPORT
AGAINST
ABSTAIN"""
    
    response = _ollama_generate(prompt).strip().upper()
    if "SUPPORT" in response:
        return "SUPPORT"
    elif "AGAINST" in response:
        return "AGAINST"
    else:
        # Default to neutral or abstain if parsing goes weird
        return "ABSTAIN"
