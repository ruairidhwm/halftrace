"""Convert provider message logs into halftrace Trajectories.

Most agent developers already have logs of their agent's conversations
— in Anthropic `messages.create()` payload shape, OpenAI chat
completions shape, or similar. This module converts those payloads to
halftrace `Trajectory` objects so the existing probes and shape
classifier can be applied without re-running anything against an API.

The functions are lossy in the sense that information not represented
in the halftrace schema (e.g. usage stats, response IDs, internal
provider fields) is preserved on `trajectory.metadata` under a
provider-specific key when present, but only the structural turns,
tool calls, and tool results are mapped to first-class trajectory data.
"""

from __future__ import annotations

import json
from typing import Any, cast

from halftrace.trajectory import ToolCall, ToolResult, Trajectory


def from_anthropic_messages(payload: dict[str, Any]) -> Trajectory:
    """Convert an Anthropic `messages.create()`-shaped payload to a Trajectory.

    Accepts the same dict shape you'd pass to
    `client.messages.create(**payload)`: a top-level `system` field
    (string or list of text blocks), a `messages` list alternating
    user/assistant, with `tool_use` / `tool_result` blocks inside content.
    An optional `metadata` field on the payload is copied verbatim to
    the trajectory's metadata.
    """
    trajectory = Trajectory(
        task_id=_str_or_none(payload.get("task_id")),
        model=_str_or_none(payload.get("model")),
        metadata=_dict_or_empty(payload.get("metadata")),
    )

    system = payload.get("system")
    if isinstance(system, str):
        trajectory.add_turn("system", content=system)
    elif isinstance(system, list):
        text = _join_anthropic_text_blocks(cast("list[Any]", system))
        if text:
            trajectory.add_turn("system", content=text)

    # Tool calls discovered in assistant turns; the matching `tool_result`
    # blocks (which carry `tool_use_id` but no tool name) borrow the name
    # back from this map.
    tool_name_by_id: dict[str, str] = {}

    raw_messages: Any = payload.get("messages")
    if not isinstance(raw_messages, list):
        return trajectory
    for raw_msg in cast("list[Any]", raw_messages):
        if not isinstance(raw_msg, dict):
            continue
        msg = cast("dict[str, Any]", raw_msg)
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            _ingest_anthropic_user(trajectory, content, tool_name_by_id)
        elif role == "assistant":
            _ingest_anthropic_assistant(trajectory, content, tool_name_by_id)

    return trajectory


def from_openai_messages(payload: dict[str, Any]) -> Trajectory:
    """Convert an OpenAI chat-completions-shaped payload to a Trajectory.

    Accepts a dict with a `messages` list containing entries with
    `role`, `content`, and (for assistant turns) `tool_calls` whose
    `function.arguments` is a JSON string. Tool results arrive as
    `role: "tool"` entries with a `tool_call_id`.
    """
    trajectory = Trajectory(
        task_id=_str_or_none(payload.get("task_id")),
        model=_str_or_none(payload.get("model")),
        metadata=_dict_or_empty(payload.get("metadata")),
    )

    tool_name_by_id: dict[str, str] = {}

    raw_messages: Any = payload.get("messages")
    if not isinstance(raw_messages, list):
        return trajectory
    for raw_msg in cast("list[Any]", raw_messages):
        if not isinstance(raw_msg, dict):
            continue
        msg = cast("dict[str, Any]", raw_msg)
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            text = _openai_content_text(content)
            if text:
                trajectory.add_turn("system", content=text)
        elif role == "user":
            text = _openai_content_text(content)
            if text:
                trajectory.add_turn("user", content=text)
        elif role == "assistant":
            _ingest_openai_assistant(trajectory, msg, tool_name_by_id)
        elif role == "tool":
            tc_id = _str_or_empty(msg.get("tool_call_id"))
            result = _openai_content_text(content)
            trajectory.add_turn(
                "tool",
                tool_results=[
                    ToolResult(
                        tool_call_id=tc_id,
                        name=tool_name_by_id.get(tc_id, ""),
                        result=result,
                    )
                ],
            )

    return trajectory


def _ingest_anthropic_user(
    trajectory: Trajectory,
    content: Any,
    tool_name_by_id: dict[str, str],
) -> None:
    if isinstance(content, str):
        trajectory.add_turn("user", content=content)
        return
    if not isinstance(content, list):
        return

    text_parts: list[str] = []
    tool_results: list[ToolResult] = []
    for raw_block in cast("list[Any]", content):
        if not isinstance(raw_block, dict):
            continue
        block = cast("dict[str, Any]", raw_block)
        btype = block.get("type")
        if btype == "text":
            text_value = block.get("text", "")
            if isinstance(text_value, str):
                text_parts.append(text_value)
        elif btype == "tool_result":
            tc_id = _str_or_empty(block.get("tool_use_id"))
            raw_result = block.get("content", "")
            result_str = _anthropic_tool_result_text(raw_result)
            tool_results.append(
                ToolResult(
                    tool_call_id=tc_id,
                    name=tool_name_by_id.get(tc_id, ""),
                    result=result_str,
                    is_error=bool(block.get("is_error", False)),
                )
            )

    if tool_results:
        trajectory.add_turn("tool", tool_results=tool_results)
    if text_parts:
        trajectory.add_turn("user", content="\n".join(text_parts))


def _ingest_anthropic_assistant(
    trajectory: Trajectory,
    content: Any,
    tool_name_by_id: dict[str, str],
) -> None:
    if isinstance(content, str):
        trajectory.add_turn("assistant", content=content)
        return
    if not isinstance(content, list):
        return

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for raw_block in cast("list[Any]", content):
        if not isinstance(raw_block, dict):
            continue
        block = cast("dict[str, Any]", raw_block)
        btype = block.get("type")
        if btype == "text":
            text_value = block.get("text", "")
            if isinstance(text_value, str):
                text_parts.append(text_value)
        elif btype == "tool_use":
            tc_id = _str_or_empty(block.get("id"))
            name = _str_or_empty(block.get("name"))
            raw_input = block.get("input", {})
            args = cast("dict[str, Any]", raw_input) if isinstance(raw_input, dict) else {}
            tool_calls.append(ToolCall(id=tc_id, name=name, args=args))
            tool_name_by_id[tc_id] = name

    text_content: str | None = "\n".join(text_parts) if text_parts else None
    trajectory.add_turn(
        "assistant",
        content=text_content,
        tool_calls=tool_calls if tool_calls else None,
    )


def _ingest_openai_assistant(
    trajectory: Trajectory,
    msg: dict[str, Any],
    tool_name_by_id: dict[str, str],
) -> None:
    raw_tool_calls = msg.get("tool_calls")
    tool_calls: list[ToolCall] = []
    if isinstance(raw_tool_calls, list):
        for raw_tc in cast("list[Any]", raw_tool_calls):
            if not isinstance(raw_tc, dict):
                continue
            tc = cast("dict[str, Any]", raw_tc)
            func = tc.get("function", {})
            if not isinstance(func, dict):
                continue
            func_dict = cast("dict[str, Any]", func)
            tc_id = _str_or_empty(tc.get("id"))
            name = _str_or_empty(func_dict.get("name"))
            args_raw = func_dict.get("arguments", "{}")
            args = _parse_openai_arguments(args_raw)
            tool_calls.append(ToolCall(id=tc_id, name=name, args=args))
            tool_name_by_id[tc_id] = name

    text = _openai_content_text(msg.get("content"))
    text_content: str | None = text if text else None
    trajectory.add_turn(
        "assistant",
        content=text_content,
        tool_calls=tool_calls if tool_calls else None,
    )


def _join_anthropic_text_blocks(blocks: list[Any]) -> str:
    text_parts: list[str] = []
    for raw in blocks:
        if not isinstance(raw, dict):
            continue
        block = cast("dict[str, Any]", raw)
        if block.get("type") == "text":
            text_value = block.get("text", "")
            if isinstance(text_value, str):
                text_parts.append(text_value)
    return "\n".join(text_parts)


def _anthropic_tool_result_text(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for item in cast("list[Any]", raw):
            if isinstance(item, dict):
                d = cast("dict[str, Any]", item)
                if d.get("type") == "text":
                    text_value = d.get("text", "")
                    if isinstance(text_value, str):
                        parts.append(text_value)
        return "\n".join(parts)
    return ""


def _openai_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for raw in cast("list[Any]", content):
            if isinstance(raw, dict):
                d = cast("dict[str, Any]", raw)
                if d.get("type") == "text":
                    text_value = d.get("text", "")
                    if isinstance(text_value, str):
                        parts.append(text_value)
        return "\n".join(parts)
    return ""


def _parse_openai_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return cast("dict[str, Any]", raw)
    if isinstance(raw, str):
        try:
            parsed: Any = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return cast("dict[str, Any]", parsed)
    return {}


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _str_or_empty(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}
