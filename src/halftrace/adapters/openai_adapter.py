"""OpenAI adapter: drive a Task through GPT and record a Trajectory.

Mirrors the Anthropic adapter's manual tool-use loop. The OpenAI API
shape differs in three load-bearing places:

- `system` is a message inside the `messages` array, not a top-level field
- Tool definitions are wrapped in `{"type": "function", "function": {...}}`
- Tool call arguments arrive as JSON strings, not pre-parsed dicts

Parallel tool use is disabled via `parallel_tool_calls=False` (vs.
Anthropic's `tool_choice.disable_parallel_tool_use=true`); the boolean
parameter name on `run_openai_task` is kept identical to the Anthropic
adapter for runner-level uniformity.
"""

from __future__ import annotations

import json
from typing import Any, cast

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion

from halftrace.tasks.base import Task, ToolSpec
from halftrace.trajectory import ToolCall, ToolResult, Trajectory

DEFAULT_OPENAI_MODEL = "gpt-4.1"
_DEFAULT_MAX_TOKENS = 4096
_DEFAULT_MAX_ITERATIONS = 500


def run_openai_task(
    task: Task,
    *,
    model: str = DEFAULT_OPENAI_MODEL,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    client: OpenAI | None = None,
    disable_parallel_tool_use: bool = False,
) -> Trajectory:
    """Run `task` against the OpenAI API and return a Trajectory.

    Behaviour matches `run_anthropic_task` from `halftrace.adapters`: the
    loop terminates when `task.is_done()` is True, the model returns a
    non-tool-call stop, or `max_iterations` is reached.
    """
    if client is None:
        client = openai.OpenAI()

    trajectory = Trajectory(task_id=task.id, model=model)
    for k, v in task.trajectory_metadata.items():
        trajectory.metadata[k] = v
    trajectory.add_turn("system", content=task.system_prompt)
    trajectory.add_turn("user", content=task.initial_user_message)

    openai_tools = _to_openai_tools(task.tool_specs)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": task.system_prompt},
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
            "messages": messages,
        }
        if openai_tools:
            create_kwargs["tools"] = openai_tools
            if disable_parallel_tool_use:
                create_kwargs["parallel_tool_calls"] = False

        response = cast(
            "ChatCompletion", client.chat.completions.create(**create_kwargs)
        )
        _accumulate_usage(usage_totals, response)
        text_content, tool_calls, tool_uses = _parse_response(response)

        finish_reason = response.choices[0].finish_reason if response.choices else None
        trajectory.add_turn(
            "assistant",
            content=text_content,
            tool_calls=tool_calls,
            metadata={"stop_reason": finish_reason},
        )

        if not tool_uses:
            break

        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": text_content,
            "tool_calls": [
                {
                    "id": tu_id,
                    "type": "function",
                    "function": {
                        "name": tu_name,
                        "arguments": json.dumps(tu_input),
                    },
                }
                for tu_id, tu_name, tu_input in tool_uses
            ],
        }
        messages.append(assistant_msg)

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
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tu_id,
                    "content": task_response.result,
                }
            )

        if task.is_done():
            break

    trajectory.metadata["usage"] = usage_totals
    return trajectory


def _accumulate_usage(totals: dict[str, int], response: ChatCompletion) -> None:
    usage = response.usage
    if usage is None:
        return
    totals["input_tokens"] += usage.prompt_tokens
    totals["output_tokens"] += usage.completion_tokens
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
        if isinstance(cached, int):
            totals["cache_read_input_tokens"] += cached
    totals["n_requests"] += 1


def _to_openai_tools(specs: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters,
            },
        }
        for s in specs
    ]


def _parse_response(
    response: ChatCompletion,
) -> tuple[str | None, list[ToolCall], list[tuple[str, str, dict[str, Any]]]]:
    if not response.choices:
        return None, [], []
    msg = response.choices[0].message
    text_content: str | None = msg.content if msg.content else None

    tool_calls: list[ToolCall] = []
    tool_uses: list[tuple[str, str, dict[str, Any]]] = []

    raw_tool_calls = msg.tool_calls or []
    for tc in raw_tool_calls:
        # The OpenAI SDK exposes tool_calls as a discriminated union; non-function
        # call types (rare) get skipped.
        if getattr(tc, "type", None) != "function":
            continue
        func = getattr(tc, "function", None)
        if func is None:
            continue
        raw_args = getattr(func, "arguments", "") or ""
        try:
            parsed: Any = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            parsed = {}
        args = cast("dict[str, Any]", parsed) if isinstance(parsed, dict) else {}
        name = getattr(func, "name", "") or ""
        tc_id = getattr(tc, "id", "") or ""
        tool_calls.append(ToolCall(id=tc_id, name=name, args=args))
        tool_uses.append((tc_id, name, args))

    return text_content, tool_calls, tool_uses
