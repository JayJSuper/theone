"""L7 · dialogue state machine — multi-turn conversation state with provenance.

Tracks the turn history and each turn's route + whether it produced a verifiable
credential, so a multi-turn session keeps an auditable record of which answers were
engine-verified and which came (unverified) from a mounted LLM. Pure state management;
no model calls here.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from theone.layer7_planning.tool_router import ToolRouter, Route, Intent


class DialogueState(str, Enum):
    IDLE = "idle"
    ROUTED = "routed"
    ANSWERED = "answered"


@dataclass
class Turn:
    index: int
    user_text: str
    route: Route
    answer: Optional[str] = None
    verified: bool = False


@dataclass
class DialogueSession:
    session_id: str
    turns: List[Turn] = field(default_factory=list)
    state: DialogueState = DialogueState.IDLE

    @property
    def verified_ratio(self) -> float:
        if not self.turns:
            return 0.0
        return sum(1 for t in self.turns if t.verified) / len(self.turns)


class DialogueStateMachine:
    def __init__(self, session_id: str = "session") -> None:
        self.session = DialogueSession(session_id=session_id)
        self.router = ToolRouter()

    def receive(self, user_text: str) -> Turn:
        """Accept a user message; route it; advance state to ROUTED."""
        route = self.router.route(user_text)
        turn = Turn(index=len(self.session.turns), user_text=user_text, route=route)
        self.session.turns.append(turn)
        self.session.state = DialogueState.ROUTED
        return turn

    def record_answer(self, answer: str, verified: bool) -> None:
        """Attach an answer to the latest turn; advance state to ANSWERED."""
        if not self.session.turns:
            raise RuntimeError("no turn to answer")
        turn = self.session.turns[-1]
        turn.answer = answer
        turn.verified = verified
        self.session.state = DialogueState.ANSWERED

    def history(self) -> List[Turn]:
        return list(self.session.turns)


__all__ = ["DialogueState", "Turn", "DialogueSession", "DialogueStateMachine"]
