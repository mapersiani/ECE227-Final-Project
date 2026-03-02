"""
Agent dataclass for network nodes.

Holds persona prompt (used by LLM), opinion text, and persona drift history.
Both opinion AND persona evolve over simulation steps as agents interact with neighbors.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Agent:
    """
    One node in the opinion network.

    - persona_prompt: starts as the nodes.json character description; drifts each step
      as the agent is influenced by its neighbors' worldviews
    - initial_persona: frozen copy of the original persona for drift analysis
    - current_opinion: the agent's current stated position on the topic
    - persona_history: list of persona snapshots at t=0, 1, 2, ...
    - opinion_history: list of opinion snapshots at t=0, 1, 2, ...
    """

    node_id: int
    persona_prompt: str          # mutable — drifts with social influence
    initial_opinion: str
    is_bot: bool = False

    # Runtime fields (not constructor args)
    current_opinion: str = field(init=False)
    initial_persona: str = field(init=False)   # frozen copy
    persona_history: List[str] = field(default_factory=list)
    opinion_history: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.current_opinion = self.initial_opinion
        self.initial_persona = self.persona_prompt   # snapshot before any drift
        # Record t=0 state
        self.persona_history = [self.persona_prompt]
        self.opinion_history = [self.initial_opinion]

    def update_opinion(self, new_opinion: str) -> None:
        """Replace current opinion; append to history."""
        self.current_opinion = new_opinion
        self.opinion_history.append(new_opinion)

    def update_persona(self, new_persona: str) -> None:
        """
        Replace the working persona prompt with a socially-influenced version.
        The new prompt is used for all future LLM opinion calls.
        Appended to persona_history for drift analysis.
        """
        self.persona_prompt = new_persona
        self.persona_history.append(new_persona)

    def persona_drift_count(self) -> int:
        """Number of times persona has been updated (0 = never drifted)."""
        return len(self.persona_history) - 1

    def reset(self) -> None:
        """Restore everything to initial state."""
        self.current_opinion = self.initial_opinion
        self.persona_prompt = self.initial_persona
        self.persona_history = [self.initial_persona]
        self.opinion_history = [self.initial_opinion]