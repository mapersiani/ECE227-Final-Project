"""
Agent dataclass for network nodes.

Holds persona prompt (used by LLM) and opinion text. Opinion evolves over simulation steps.
"""

from dataclasses import dataclass, field


@dataclass
class Agent:
    """
    One node in the opinion network. Has a fixed persona and evolving opinion text.
    """

    node_id: int
    persona_prompt: str
    initial_opinion: str
    current_opinion: str = field(init=False)
    is_bot: bool = False

    def __post_init__(self) -> None:
        self.current_opinion = self.initial_opinion

    def update_opinion(self, new_opinion: str) -> None:
        """Replace current opinion with LLM-generated update."""
        self.current_opinion = new_opinion

    def reset(self) -> None:
        """Restore opinion to initial value."""
        self.current_opinion = self.initial_opinion
