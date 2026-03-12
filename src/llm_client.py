"""
LLM client for opinion updates via Ollama.

Calls local Ollama API. No API key; requires Ollama running with a pulled model.
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
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
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
    Ask Ollama for an updated opinion given persona, topic, and neighbor opinions.

    Args:
        persona: Agent's persona prompt
        topic: Discussion topic (e.g. "AI Regulation")
        neighbor_opinions: List of neighbor opinion texts
        memory: Optional prior context

    Returns:
        New opinion text (1–2 sentences).
    """
    neighbor_text = "\n".join(f"- Neighbor {i + 1}: {o}" for i, o in enumerate(neighbor_opinions))
    memory_line = f"\nPrevious context or memory: {memory}\n" if memory else ""
    prompt = f"""You are simulating an agent in a social network opinion dynamics experiment.

{persona}

The topic under discussion is: {topic}

You have just read the following opinions from your neighbors:

{neighbor_text}
{memory_line}
In 1-2 concise sentences, state your updated opinion on this topic. Reflect how you are influenced by your neighbors' arguments (or lack thereof), but stay in character. Output only the opinion text, no meta-commentary."""
    return _ollama_generate(prompt)
