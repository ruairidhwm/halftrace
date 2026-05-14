"""Task protocol and supporting types.

A Task is the environment side of an agent loop: it owns the system prompt,
initial user message, tool definitions, and the logic that responds to each
tool call. The adapter (a separate layer) drives an agent through a task by
forwarding tool calls and recording every turn to a Trajectory.
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class ToolSpec(BaseModel):
    """Provider-agnostic description of a tool the agent can call.

    The `parameters` field is a JSON Schema object, matching the common
    shape across OpenAI and Anthropic tool definitions.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, Any]


class ToolResponse(BaseModel):
    """The environment's reply to a tool call, plus optional probe annotations.

    `annotations` is merged into the metadata of the tool turn recorded in
    the trajectory, where probes read it (e.g. `state_amnesia` looks at
    `metadata["state_amnesia"]`).
    """

    model_config = ConfigDict(extra="forbid")

    result: str
    is_error: bool = False
    annotations: dict[str, Any] = Field(default_factory=lambda: {})


class Task(Protocol):
    """A self-contained agentic task.

    Adapters consume Task instances structurally: read the prompts and
    tool specs to set up the agent, forward tool calls to `handle_tool_call`,
    and stop when `is_done` returns True. `trajectory_metadata` is merged
    into the resulting Trajectory's metadata so probes (e.g. instruction_decay)
    can find their configuration.
    """

    id: str
    system_prompt: str
    initial_user_message: str
    tool_specs: list[ToolSpec]
    trajectory_metadata: dict[str, Any]

    def handle_tool_call(self, name: str, args: dict[str, Any]) -> ToolResponse: ...

    def is_done(self) -> bool: ...
