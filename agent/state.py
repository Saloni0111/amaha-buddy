"""
Structured conversation state.

Kept explicit and separate from raw chat history so the orchestrator's
decisions (ask another question vs. route) are testable and legible,
rather than buried inside whatever the LLM feels like doing with a
big blob of chat history.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class ConversationState:
    history: List[Dict[str, str]] = field(default_factory=list)  # [{"role": "user"/"assistant", "content": str}]
    topic: Optional[str] = None            # e.g. "work stress", "grief", "relationship"
    severity: Optional[str] = None         # "low" | "moderate" | "high"
    turn_count: int = 0
    ready_to_route: bool = False
    crisis_flag: bool = False
    routed_category: Optional[str] = None  # set once routing has happened

    def add_message(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})

    def as_dict(self) -> dict:
        return {
            "topic": self.topic,
            "severity": self.severity,
            "turn_count": self.turn_count,
            "ready_to_route": self.ready_to_route,
            "crisis_flag": self.crisis_flag,
            "routed_category": self.routed_category,
        }
