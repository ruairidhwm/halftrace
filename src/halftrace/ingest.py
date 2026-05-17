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


def from_langsmith_run(run: dict[str, Any]) -> Trajectory:
    """Convert a LangSmith trace run to a halftrace Trajectory.

    LangSmith records an agent invocation as a tree of `run` objects:
    a root `chain` run with nested `llm`, `tool`, and further `chain`
    runs as children. Each LLM run carries the full conversation prefix
    in its `inputs.messages`; the final LLM run in chronological order
    therefore contains everything the agent saw up to its last action.

    The conversion strategy:

    1. Walk the run tree (depth-first, ordered by `start_time` when
       present) and collect every LLM run.
    2. If any LLM runs are present, use the **last** LLM run's
       `inputs.messages` as the conversation prefix (OpenAI-format),
       then append a final assistant turn synthesised from the last
       LLM run's `outputs` (text content and `tool_calls` if any).
    3. If no LLM runs are present (e.g. a trace that's purely chain
       coordination), fall back to mining `inputs.messages` from the
       root run, returning whatever is there.

    The function is tolerant of LangSmith schema variation: missing or
    differently-named fields produce empty turns rather than raising.
    The top-level `id`, `name`, `start_time`, and `end_time` are copied
    to `trajectory.metadata["langsmith"]` for downstream inspection.
    """
    llm_runs: list[dict[str, Any]] = []
    _collect_llm_runs(run, llm_runs)

    if llm_runs:
        last = llm_runs[-1]
        raw_inputs = last.get("inputs")
        messages: list[Any] = []
        if isinstance(raw_inputs, dict):
            raw_messages = cast("dict[str, Any]", raw_inputs).get("messages")
            if isinstance(raw_messages, list):
                messages = list(cast("list[Any]", raw_messages))
        # Build a payload accepted by from_openai_messages, plus an
        # appended assistant turn synthesised from the last LLM's outputs.
        final_assistant = _langsmith_synthesise_final_assistant(last)
        if final_assistant is not None:
            messages.append(final_assistant)
        payload = {
            "task_id": _str_or_none(run.get("id")),
            "model": _str_or_none(_langsmith_model_name(last)),
            "messages": messages,
        }
    else:
        raw_inputs = run.get("inputs")
        messages = []
        if isinstance(raw_inputs, dict):
            raw_messages = cast("dict[str, Any]", raw_inputs).get("messages")
            if isinstance(raw_messages, list):
                messages = list(cast("list[Any]", raw_messages))
        payload = {
            "task_id": _str_or_none(run.get("id")),
            "model": _str_or_none(_langsmith_model_name(run)),
            "messages": messages,
        }

    trajectory = from_openai_messages(payload)

    # Preserve enough of the LangSmith envelope that callers can
    # cross-reference back to the source trace.
    langsmith_meta: dict[str, Any] = {}
    for key in ("id", "name", "start_time", "end_time"):
        value = run.get(key)
        if isinstance(value, str):
            langsmith_meta[key] = value
    langsmith_meta["n_llm_runs"] = len(llm_runs)
    trajectory.metadata["langsmith"] = langsmith_meta
    return trajectory


def _collect_llm_runs(node: Any, acc: list[dict[str, Any]]) -> None:
    """Depth-first walk that collects every run whose run_type is 'llm'."""
    if not isinstance(node, dict):
        return
    typed = cast("dict[str, Any]", node)
    run_type = typed.get("run_type")
    if run_type in ("llm", "chat_model"):
        acc.append(typed)
    for key in ("child_runs", "children"):
        raw_children = typed.get(key)
        if isinstance(raw_children, list):
            for child in cast("list[Any]", raw_children):
                _collect_llm_runs(child, acc)


def _langsmith_model_name(run: dict[str, Any]) -> str | None:
    """Extract a model name from a LangSmith LLM run, if present."""
    # LangSmith places model identity in several spots depending on the
    # integration. Check the most common ones.
    raw_extra = run.get("extra")
    if isinstance(raw_extra, dict):
        extra = cast("dict[str, Any]", raw_extra)
        invocation_params = extra.get("invocation_params")
        if isinstance(invocation_params, dict):
            ip = cast("dict[str, Any]", invocation_params)
            for key in ("model", "model_name", "model_id"):
                value = ip.get(key)
                if isinstance(value, str):
                    return value
    raw_serialized = run.get("serialized")
    if isinstance(raw_serialized, dict):
        serialized = cast("dict[str, Any]", raw_serialized)
        kwargs = serialized.get("kwargs")
        if isinstance(kwargs, dict):
            k = cast("dict[str, Any]", kwargs)
            for key in ("model", "model_name"):
                value = k.get(key)
                if isinstance(value, str):
                    return value
    return None


def _langsmith_synthesise_final_assistant(run: dict[str, Any]) -> dict[str, Any] | None:
    """Build a final assistant message dict from a LangSmith LLM run's outputs.

    LangSmith stores LLM outputs in several shapes. The most common
    is `outputs.generations[0][0].message` or `outputs.generations[0].message`
    with the message in roughly OpenAI format. We extract `content`
    and `tool_calls` if present, returning a dict appropriate for
    appending to a `messages` list consumed by from_openai_messages.
    """
    raw_outputs = run.get("outputs")
    if not isinstance(raw_outputs, dict):
        return None
    outputs = cast("dict[str, Any]", raw_outputs)
    message = _walk_for_message(outputs)
    if message is None:
        return None
    text = _str_or_empty(message.get("content"))
    tool_calls = message.get("tool_calls")
    msg: dict[str, Any] = {"role": "assistant"}
    if text:
        msg["content"] = text
    if isinstance(tool_calls, list) and tool_calls:
        msg["tool_calls"] = tool_calls
    if "content" not in msg and "tool_calls" not in msg:
        return None
    return msg


def _walk_for_message(node: Any) -> dict[str, Any] | None:
    """Find the first dict that looks like an OpenAI-format assistant message."""
    if isinstance(node, dict):
        typed = cast("dict[str, Any]", node)
        if "content" in typed or "tool_calls" in typed:
            # Looks message-like; only accept if role suggests assistant or
            # there is no role at all (LangSmith often omits it).
            role = typed.get("role")
            if role is None or role == "assistant":
                return typed
        for value in typed.values():
            found = _walk_for_message(value)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in cast("list[Any]", node):
            found = _walk_for_message(item)
            if found is not None:
                return found
    return None


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
