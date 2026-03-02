"""
LLM client for opinion and persona updates via Ollama.

Two separate prompt types:
  - get_updated_opinion: agent reads neighbor opinions, outputs new stance (1-2 sentences)
  - get_updated_persona: agent's self-concept drifts subtly from social exposure,
    always anchored to the original nodes.json persona so the character never
    drifts beyond recognition across multiple steps
"""

import json
import urllib.error
import urllib.request
from typing import Sequence

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL


def _ollama_generate(prompt: str) -> str:
    """
    Send prompt to Ollama /api/generate. Returns model response text.
    Raises RuntimeError if Ollama is not running.
    """
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    body = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
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


def get_updated_opinion(
    persona: str,
    topic: str,
    neighbor_opinions: Sequence[str],
    memory: str = "",
) -> str:
    """
    Ask the LLM for an updated opinion given persona, topic, and neighbor opinions.

    Uses the agent's CURRENT persona prompt (which may have already drifted),
    so opinion updates naturally reflect persona drift.

    Returns:
        New opinion text (1-2 sentences).
    """
    neighbor_text = "\n".join(f"- {o}" for o in neighbor_opinions)
    memory_line = f"\nPrevious context: {memory}\n" if memory else ""
    prompt = f"""You are simulating an agent in a social network opinion dynamics experiment.

{persona}

The topic under discussion is: {topic}

You have just read the following opinions from your social connections:

{neighbor_text}
{memory_line}
In 1-2 concise sentences, state your updated opinion on this topic. Stay true to your character and values, but genuinely engage with what you have heard. Output only the opinion text, no meta-commentary."""
    return _ollama_generate(prompt)


def get_updated_persona(
    original_persona: str,
    current_persona: str,
    neighbor_personas: Sequence[str],
    neighbor_opinions: Sequence[str],
    topic: str,
) -> str:
    """
    Compute a persona drift update, always anchored to the original nodes.json character.

    The original_persona is the immutable nodes.json description and acts as a hard
    anchor. The agent's professional role, background, and core values must be
    preserved exactly. Only their emphasis, openness, or rhetorical framing may shift
    from social exposure.

    Using original_persona (not current_persona) as the baseline prevents drift from
    compounding across steps and the character losing its identity over time.

    Args:
        original_persona: The frozen nodes.json persona — the immutable anchor.
        current_persona:  The most recent drifted persona (provided for context only).
        neighbor_personas: Neighbors' current persona descriptions.
        neighbor_opinions: Neighbors' current opinion texts.
        topic: Simulation topic.

    Returns:
        Updated persona description (2-4 sentences), grounded in original_persona.
    """
    neighbor_voices = "\n".join(
        f"- [{i+1}] Worldview: {p[:200]}... | Recent opinion: {o}"
        for i, (p, o) in enumerate(zip(neighbor_personas, neighbor_opinions))
    )
    prompt = f"""You are modeling subtle identity drift in a social network simulation.

An agent has the following fixed background and professional identity. This cannot change:

ORIGINAL PERSONA (immutable anchor):
{original_persona}

After recent social interactions, their working perspective has shifted slightly from the original:

CURRENT PERSONA (for context only, may have drifted):
{current_persona}

Their social connections hold these worldviews and recent opinions on "{topic}":

NEIGHBORS:
{neighbor_voices}

Write a revised persona description (2-4 sentences) that:
- Keeps the professional role, background, and core values IDENTICAL to the ORIGINAL PERSONA
- Reflects only subtle shifts in emphasis, rhetorical openness, or framing from social exposure
- Stays realistic: people do not flip their identity entirely; they nuance, accommodate, or entrench
- Is written in third person, same style as the original

Output ONLY the revised persona description, no preamble or explanation."""
    return _ollama_generate(prompt)