"""Anthropic adapter: drive a Task through Claude and record a Trajectory.

Wraps the SDK manual tool-use loop in a function that bridges Task to Trajectory:
- task.system_prompt / initial_user_message / tool_specs configure the API call
- each tool_use block is dispatched to task.handle_tool_call
- every model turn and tool turn is recorded, with probe annotations attached

Defaults reflect the context-rot study in HYPOTHESES.md: model
`claude-opus-4-7`, no thinking (clean baseline), prompt caching enabled.
Sampling parameters (`temperature`, `top_p`, `top_k`) and `budget_tokens`
are deliberately not exposed because they are removed on Opus 4.7.
"""

from __future__ import annotations

from typing import Any, cast

import anthropic
from anthropic import Anthropic
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

from halftrace.tasks.base import Task, ToolSpec
from halftrace.trajectory import ToolCall, ToolResult, Trajectory

DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-7"
_DEFAULT_MAX_TOKENS = 4096
_DEFAULT_MAX_ITERATIONS = 500


def run_anthropic_task(
    task: Task,
    *,
    model: str = DEFAULT_ANTHROPIC_MODEL,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    client: Anthropic | None = None,
    thinking: dict[str, Any] | None = None,
    cache_prompt: bool = True,
) -> Trajectory:
    """Run `task` against the Anthropic API and return a Trajectory.

    The loop terminates when `task.is_done()` returns True, the model emits
    `end_turn` (or any non-`tool_use` stop) without further tool calls, or
    `max_iterations` is reached. Every assistant and tool turn is recorded;
    probe annotations attached to tool responses propagate to turn metadata.

    When `cache_prompt=True` the system prompt and last tool spec are marked
    with `cache_control` so the prefix is reused across iterations of one
    trajectory. The minimum cacheable prefix is model-dependent (4096 tokens
    on Opus 4.7); short prefixes silently won't cache but the marker is
    harmless.
    """
    if client is None:
        client = anthropic.Anthropic()

    trajectory = Trajectory(task_id=task.id, model=model)
    trajectory.add_turn("system", content=task.system_prompt)
    trajectory.add_turn("user", content=task.initial_user_message)

    anthropic_tools = _to_anthropic_tools(task.tool_specs, cache=cache_prompt)
    system_param: str | list[dict[str, Any]]
    if cache_prompt:
        system_param = [
            {
                "type": "text",
                "text": task.system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        system_param = task.system_prompt

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": task.initial_user_message}
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
        if thinking is not None:
            create_kwargs["thinking"] = thinking

        response = cast("Message", client.messages.create(**create_kwargs))
        _accumulate_usage(usage_totals, response.usage)
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

    trajectory.metadata["usage"] = usage_totals
    return trajectory


def _accumulate_usage(totals: dict[str, int], usage: Usage) -> None:
    totals["input_tokens"] += usage.input_tokens
    totals["output_tokens"] += usage.output_tokens
    totals["cache_creation_input_tokens"] += usage.cache_creation_input_tokens or 0
    totals["cache_read_input_tokens"] += usage.cache_read_input_tokens or 0
    totals["n_requests"] += 1


def _to_anthropic_tools(specs: list[ToolSpec], *, cache: bool) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = [
        {"name": s.name, "description": s.description, "input_schema": s.parameters}
        for s in specs
    ]
    if cache and tools:
        tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
    return tools


def _parse_response(
    response: Message,
) -> tuple[str | None, list[ToolCall], list[tuple[str, str, dict[str, Any]]]]:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    tool_uses: list[tuple[str, str, dict[str, Any]]] = []

    for block in response.content:
        if isinstance(block, TextBlock):
            text_parts.append(block.text)
        elif isinstance(block, ToolUseBlock):
            args = cast("dict[str, Any]", block.input)
            tool_calls.append(ToolCall(id=block.id, name=block.name, args=args))
            tool_uses.append((block.id, block.name, args))

    text_content = "\n".join(text_parts) if text_parts else None
    return text_content, tool_calls, tool_uses


def _to_param_blocks(content: list[Any]) -> list[dict[str, Any]]:
    """Round-trip response content to API-shaped param blocks for re-sending."""
    blocks: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, TextBlock):
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolUseBlock):
            input_val = cast("dict[str, Any]", block.input)
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": input_val,
                }
            )
    return blocks
