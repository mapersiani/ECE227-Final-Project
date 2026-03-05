"""Canonical run matrix for the final government-environment experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.config import TOPIC

GraphType = Literal["er", "rgg_long_range"]
BotMode = Literal["off", "on"]


@dataclass(frozen=True)
class RunSpec:
    """One experiment condition in the 2x2 matrix."""

    run_id: str
    graph: GraphType
    bot: BotMode


RUN_MATRIX: list[RunSpec] = [
    RunSpec(run_id="ER_no_bot", graph="er", bot="off"),
    RunSpec(run_id="ER_bot", graph="er", bot="on"),
    RunSpec(run_id="RGGLR_no_bot", graph="rgg_long_range", bot="off"),
    RunSpec(run_id="RGGLR_bot", graph="rgg_long_range", bot="on"),
]
