"""LLM client for opinion updates (Gemini API)."""

import os
from typing import Sequence

from src.config import GOOGLE_API_KEY


def _get_client():
    """Lazy import to avoid loading if not used."""
    import google.generativeai as genai

    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY not set. Copy .env.example to .env and add your key."
        )
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai.GenerativeModel("gemini-1.5-flash")


def get_updated_opinion(
    persona: str,
    topic: str,
    neighbor_opinions: Sequence[str],
    memory: str = "",
) -> str:
    """
    Query Gemini to produce an updated opinion given persona, topic, and neighbors' opinions.

    Args:
        persona: The agent's persona prompt
        topic: The controversial topic (e.g. "AI Regulation")
        neighbor_opinions: List of neighbor opinion texts
        memory: Optional memory / previous context

    Returns:
        The agent's new opinion text
    """
    genai_client = _get_client()

    neighbor_text = "\n".join(
        f"- Neighbor {i + 1}: {o}" for i, o in enumerate(neighbor_opinions)
    )

    prompt = f"""You are simulating an agent in a social network opinion dynamics experiment.

{persona}

The topic under discussion is: {topic}

You have just read the following opinions from your neighbors:

{neighbor_text}

{f"Previous context or memory: {memory}" if memory else ""}

In 1–2 concise sentences, state your updated opinion on this topic. Reflect how you are influenced by your neighbors' arguments (or lack thereof), but stay in character. Output only the opinion text, no meta-commentary."""

    response = genai_client.generate_content(prompt)
    return (response.text or "").strip()
