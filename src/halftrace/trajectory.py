"""Trajectory schema for halftrace.

A Trajectory is the recorded sequence of an agent's turns through a task.
It is provider-agnostic: trajectories can be constructed from OpenAI messages,
Anthropic messages, or built turn-by-turn from any agent loop.

Probes consume Trajectories and produce diagnostic scores.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    """A single tool invocation emitted by the assistant."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    args: dict[str, Any]


class ToolResult(BaseModel):
    """The result of executing a tool call, returned to the agent."""

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    name: str
    result: str
    is_error: bool = False


class Turn(BaseModel):
    """A single turn in an agent trajectory.

    A turn has a role, optional text content, and optional tool calls or results.
    Turn indices are zero-based and assigned in the order turns are added.
    """

    model_config = ConfigDict(extra="forbid")

    index: int
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=lambda: [])
    tool_results: list[ToolResult] = Field(default_factory=lambda: [])
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Trajectory(BaseModel):
    """The recorded sequence of an agent's turns through a task.

    Trajectories are append-only: once added, turns are not mutated.
    Build them by calling `add_turn` repeatedly, or construct from
    provider-specific message formats via the `from_*` class methods.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str | None = None
    model: str | None = None
    turns: list[Turn] = Field(default_factory=lambda: [])
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_turn(
        self,
        role: Role,
        content: str | None = None,
        tool_calls: list[ToolCall] | None = None,
        tool_results: list[ToolResult] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Turn:
        """Append a turn to the trajectory. Returns the created turn."""
        turn = Turn(
            index=len(self.turns),
            role=role,
            content=content,
            tool_calls=tool_calls or [],
            tool_results=tool_results or [],
            metadata=metadata or {},
        )
        self.turns.append(turn)
        return turn

    def tool_calls(self) -> list[tuple[int, ToolCall]]:
        """All tool calls in the trajectory, paired with their turn index."""
        return [(turn.index, tc) for turn in self.turns for tc in turn.tool_calls]

    def tool_results(self) -> list[tuple[int, ToolResult]]:
        """All tool results in the trajectory, paired with their turn index."""
        return [(turn.index, tr) for turn in self.turns for tr in turn.tool_results]

    def token_count_estimate(self) -> int:
        """Rough token-count estimate (chars / 4).

        Sufficient for relative comparisons across turns. For accurate counts,
        use tiktoken or the provider SDKs directly.
        """
        total = 0
        for turn in self.turns:
            if turn.content is not None:
                total += len(turn.content)
            for tc in turn.tool_calls:
                total += len(tc.name) + len(json.dumps(tc.args))
            for tr in turn.tool_results:
                total += len(tr.result)
        return total // 4

    def to_jsonl_line(self) -> str:
        """Serialise to a single JSONL line."""
        return self.model_dump_json()

    @classmethod
    def from_jsonl_line(cls, line: str) -> Trajectory:
        """Deserialise from a single JSONL line."""
        return cls.model_validate_json(line)
