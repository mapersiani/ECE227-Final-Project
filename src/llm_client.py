# src/llm_client.py
"""
LLM client for opinion updates via Ollama.

Upgrades:
- retries/backoff for transient failures
- returns "" on failure (caller can keep old opinion)
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Sequence

from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL


def _ollama_generate(prompt: str, timeout: int = 120, retries: int = 3, backoff_s: float = 1.5) -> str:
    """
    Send prompt to Ollama /api/generate. Returns model response text.
    On failure, returns "" (so simulation can continue).
    """
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    body = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            return (data.get("response") or "").strip()
        except urllib.error.URLError as e:
            last_err = e
            msg = str(e).lower()
            # If Ollama isn't running, don't spam retries forever.
            if "connection refused" in msg or "localhost" in msg:
                return ""
            if attempt < retries:
                time.sleep(backoff_s * attempt)
            else:
                return ""
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff_s * attempt)
            else:
                return ""

    # should never reach
    _ = last_err
    return ""


def get_updated_opinion(
    persona: str,
    topic: str,
    neighbor_opinions: Sequence[str],
    memory: str = "",
) -> str:
    """
    Ask Ollama for an updated opinion given persona, topic, and neighbor opinions.

    Returns:
        New opinion text (1–2 sentences). May be "" if model call fails.
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
