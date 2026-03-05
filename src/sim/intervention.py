"""Intervention simulation adapters for integration branch."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.intervention import run_with_bot


def run_intervention(
    topic: str,
    steps: int,
    bot_post_prob: float,
    seed: int,
    edge_prob: float,
    log_path: Optional[str | Path] = None,
) -> tuple[list[float], list[dict[str, int]]]:
    """Compatibility wrapper for disinformation intervention runs."""
    return run_with_bot(
        topic=topic,
        steps=steps,
        bot_post_prob=bot_post_prob,
        seed=seed,
        edge_prob=edge_prob,
        log_path=log_path,
    )
