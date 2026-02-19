"""Agent representation with persona and opinion text."""

from dataclasses import dataclass, field


@dataclass
class Agent:
    """An LLM agent with a persona and evolving opinion."""

    node_id: int
    persona_prompt: str
    initial_opinion: str
    current_opinion: str = field(init=False)
    is_bot: bool = False

    def __post_init__(self) -> None:
        self.current_opinion = self.initial_opinion

    def update_opinion(self, new_opinion: str) -> None:
        """Update the agent's current opinion."""
        self.current_opinion = new_opinion

    def reset(self) -> None:
        """Reset opinion to initial state."""
        self.current_opinion = self.initial_opinion
