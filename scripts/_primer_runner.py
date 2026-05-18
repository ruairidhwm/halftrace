"""Shared helper for the mechanism scripts.

Each mechanism investigation (mechanism_test, _control, _minimal, _zero,
_long, _content_rule, _marker_rule, _haiku) prepends a different primer
message list to the API call before running the real task. The bulk of
each script is identical: the API loop, tool-result handling, usage
tracking, and scoring across the four-probe set. This module factors that
out so each mechanism script is just a configuration of primer + model +
output path.

The mechanism scripts deliberately reach into the adapter's private
helpers (`_to_anthropic_tools`, `_parse_response`, `_to_param_blocks`,
`_accumulate_usage`) because the primer is injected at the message-list
level, not at the task level — there is no public Task hook for that.
Promoting these helpers to public API would commit halftrace to a stable
adapter-internals surface for the sake of one experimental workflow,
which is the wrong trade. Strict pyright is disabled on `scripts/` for
exactly this reason (see `pyproject.toml`).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import anthropic
from anthropic.types import Message, Usage

from halftrace.adapters.anthropic_adapter import (
    _accumulate_usage,
    _parse_response,
    _to_anthropic_tools,
    _to_param_blocks,
)
from halftrace.probes import (
    Score,
    instruction_decay,
    narration_substitution,
    state_amnesia,
    tool_repetition,
)
from halftrace.trajectory import ToolResult, Trajectory

PROBES = {
    "state_amnesia": state_amnesia,
    "instruction_decay": instruction_decay,
    "tool_repetition": tool_repetition,
    "narration_substitution": narration_substitution,
}


def run_with_primer(
    task: Any,
    *,
    primer: list[dict[str, Any]],
    model: str,
    client: anthropic.Anthropic,
    max_tokens: int = 4096,
    max_iterations: int = 500,
    disable_parallel_tool_use: bool = True,
) -> Trajectory:
    """Run a task with the given primer messages prepended to the conversation.

    The returned Trajectory contains only the real-task turns (system,
    initial user message, and every assistant/tool turn produced by the
    agent during the loop). Primer messages are sent to the API but stored
    on `trajectory.metadata["primer_messages"]` for reference rather than
    added as scored turns — keeping the per-N rep count clean against
    pilot baselines.

    Usage totals across all API requests are accumulated into
    `trajectory.metadata["usage_totals"]`. The task's
    `trajectory_metadata` is merged in first, so probes that read from
    `Trajectory.metadata` (instruction_decay, premature_termination)
    score correctly.
    """
    trajectory = Trajectory(task_id=task.id, model=model)
    for k, v in task.trajectory_metadata.items():
        trajectory.metadata[k] = v
    trajectory.add_turn("system", content=task.system_prompt)
    trajectory.add_turn("user", content=task.initial_user_message)

    anthropic_tools = _to_anthropic_tools(task.tool_specs, cache=True)
    system_param: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": task.system_prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    messages: list[dict[str, Any]] = [
        *primer,
        {"role": "user", "content": task.initial_user_message},
    ]
    usage_totals: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "n_requests": 0,
    }

    for _ in range(max_iterations):
        create_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_param,
            "messages": messages,
        }
        if anthropic_tools:
            create_kwargs["tools"] = anthropic_tools
            if disable_parallel_tool_use:
                create_kwargs["tool_choice"] = {
                    "type": "auto",
                    "disable_parallel_tool_use": True,
                }

        response = cast("Message", client.messages.create(**create_kwargs))
        _accumulate_usage(usage_totals, cast("Usage", response.usage))
        text_content, tool_calls, tool_uses = _parse_response(response)

        trajectory.add_turn(
            "assistant",
            content=text_content,
            tool_calls=tool_calls,
            metadata={"stop_reason": response.stop_reason},
        )

        if not tool_uses:
            break

        messages.append(
            {"role": "assistant", "content": _to_param_blocks(response.content)}
        )

        tool_result_blocks: list[dict[str, Any]] = []
        for tu_id, tu_name, tu_input in tool_uses:
            task_response = task.handle_tool_call(tu_name, tu_input)
            trajectory.add_turn(
                "tool",
                tool_results=[
                    ToolResult(
                        tool_call_id=tu_id,
                        name=tu_name,
                        result=task_response.result,
                        is_error=task_response.is_error,
                    )
                ],
                metadata=task_response.annotations,
            )
            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu_id,
                    "content": task_response.result,
                    "is_error": task_response.is_error,
                }
            )

        messages.append({"role": "user", "content": tool_result_blocks})

        if task.is_done():
            break

    trajectory.metadata["usage_totals"] = usage_totals
    trajectory.metadata["primer_messages"] = primer
    return trajectory


def score_all(trajectory: Trajectory) -> Mapping[str, Score]:
    """Score a trajectory against the four-probe set the mechanism scripts use.

    Returns a mapping from probe name to Score. Suitable for JSON-encoding
    into a results record via `s.model_dump(mode="json")`.
    """
    return {name: probe(trajectory) for name, probe in PROBES.items()}
