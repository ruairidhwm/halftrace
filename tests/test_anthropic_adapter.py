"""Tests for the Anthropic adapter.

These tests drive the adapter with a fake client; no API calls are made.
"""

from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from typing import Any, cast

from anthropic import Anthropic

from halftrace import find_and_synthesise, state_amnesia
from halftrace.adapters import run_anthropic_task


def _text_block(text: str) -> Any:
    from anthropic.types import TextBlock

    return TextBlock(type="text", text=text, citations=None)


def _tool_block(id_: str, name: str, input_: dict[str, Any]) -> Any:
    from anthropic.types import ToolUseBlock

    return ToolUseBlock(type="tool_use", id=id_, name=name, input=input_)


def _usage(input_tokens: int = 0, output_tokens: int = 0) -> Any:
    from anthropic.types import Usage

    return Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
        server_tool_use=None,
        service_tier=None,
    )


class _Response:
    def __init__(
        self, content: list[Any], stop_reason: str, usage: Any | None = None
    ) -> None:
        self.content = content
        self.stop_reason = stop_reason
        self.usage = usage if usage is not None else _usage()


class _Messages:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        snapshot = dict(kwargs)
        if "messages" in snapshot:
            snapshot["messages"] = [dict(m) for m in snapshot["messages"]]
        self.calls.append(snapshot)
        if not self.responses:
            raise RuntimeError("test fake ran out of canned responses")
        return self.responses.pop(0)


class _FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.messages = _Messages(responses)


def _client(*responses: Any) -> tuple[Anthropic, _FakeClient]:
    fake = _FakeClient(list(responses))
    return cast("Anthropic", fake), fake


class TestHappyPath:
    """End-to-end runs against the fake client."""

    def test_loop_records_every_assistant_and_tool_turn(self) -> None:
        task = find_and_synthesise(2)
        codeword = task.planted_codewords[0]
        client, _ = _client(
            _Response([_tool_block("c1", "lookup", {"topic": "topic_1"})], "tool_use"),
            _Response([_tool_block("c2", "lookup", {"topic": "topic_2"})], "tool_use"),
            _Response(
                [
                    _text_block(f"The password is {codeword}."),
                    _tool_block("c3", "submit_summary", {"summary": "done"}),
                ],
                "tool_use",
            ),
        )
        trajectory = run_anthropic_task(task, client=client)
        assert task.is_done()
        # system + user + 3 x (assistant + tool) = 8 turns
        assert len(trajectory.turns) == 8
        assert trajectory.turns[0].role == "system"
        assert trajectory.turns[1].role == "user"
        roles = [t.role for t in trajectory.turns[2:]]
        assert roles == ["assistant", "tool", "assistant", "tool", "assistant", "tool"]

    def test_state_amnesia_scores_a_correct_recall_through_the_adapter(self) -> None:
        task = find_and_synthesise(3)
        codeword = task.planted_codewords[0]
        client, _ = _client(
            _Response([_tool_block("c1", "lookup", {"topic": "topic_1"})], "tool_use"),
            _Response([_tool_block("c2", "lookup", {"topic": "topic_2"})], "tool_use"),
            _Response([_tool_block("c3", "lookup", {"topic": "topic_3"})], "tool_use"),
            _Response(
                [
                    _text_block(f"The password I remember is {codeword}."),
                    _tool_block("c4", "submit_summary", {"summary": "ok"}),
                ],
                "tool_use",
            ),
        )
        trajectory = run_anthropic_task(task, client=client)
        score = state_amnesia(trajectory)
        assert score.value == 1.0
        assert score.n_observations == 1

    def test_assistant_turn_carries_stop_reason_metadata(self) -> None:
        task = find_and_synthesise(2)
        client, _ = _client(
            _Response([_tool_block("c1", "lookup", {"topic": "topic_1"})], "tool_use"),
            _Response([_tool_block("c2", "submit_summary", {"summary": "ok"})], "tool_use"),
        )
        trajectory = run_anthropic_task(task, client=client)
        assistant_turns = [t for t in trajectory.turns if t.role == "assistant"]
        assert assistant_turns[0].metadata["stop_reason"] == "tool_use"
        assert assistant_turns[1].metadata["stop_reason"] == "tool_use"


class TestTermination:
    """The loop exits cleanly under various stop conditions."""

    def test_end_turn_without_tool_use_breaks_the_loop(self) -> None:
        task = find_and_synthesise(5)
        client, _ = _client(_Response([_text_block("I'd rather not.")], "end_turn"))
        trajectory = run_anthropic_task(task, client=client)
        assert not task.is_done()
        assert len(trajectory.turns) == 3  # system + user + assistant

    def test_max_iterations_terminates_a_runaway_agent(self) -> None:
        task = find_and_synthesise(5)
        responses = [
            _Response([_tool_block(f"c{i}", "lookup", {"topic": "topic_1"})], "tool_use")
            for i in range(20)
        ]
        client, _ = _client(*responses)
        trajectory = run_anthropic_task(task, client=client, max_iterations=3)
        assert not task.is_done()
        # 2 setup turns + 3 iterations x (assistant + tool) = 8
        assert len(trajectory.turns) == 8


class TestApiCall:
    """The kwargs we send to the API match the task and our cache config."""

    def test_tools_use_anthropic_input_schema_shape(self) -> None:
        task = find_and_synthesise(2)
        client, fake = _client(
            _Response(
                [_tool_block("c1", "submit_summary", {"summary": "done"})], "tool_use"
            )
        )
        run_anthropic_task(task, client=client)
        call = fake.messages.calls[0]
        names = {t["name"] for t in call["tools"]}
        assert names == {"lookup", "submit_summary"}
        for tool in call["tools"]:
            assert "input_schema" in tool
            assert "parameters" not in tool

    def test_caching_marks_system_and_last_tool(self) -> None:
        task = find_and_synthesise(2)
        client, fake = _client(
            _Response(
                [_tool_block("c1", "submit_summary", {"summary": "done"})], "tool_use"
            )
        )
        run_anthropic_task(task, client=client, cache_prompt=True)
        call = fake.messages.calls[0]
        assert isinstance(call["system"], list)
        assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" in call["tools"][-1]
        assert "cache_control" not in call["tools"][0]

    def test_caching_can_be_disabled(self) -> None:
        task = find_and_synthesise(2)
        client, fake = _client(
            _Response(
                [_tool_block("c1", "submit_summary", {"summary": "done"})], "tool_use"
            )
        )
        run_anthropic_task(task, client=client, cache_prompt=False)
        call = fake.messages.calls[0]
        assert isinstance(call["system"], str)
        for tool in call["tools"]:
            assert "cache_control" not in tool

    def test_default_model_is_opus_4_7(self) -> None:
        task = find_and_synthesise(2)
        client, fake = _client(
            _Response(
                [_tool_block("c1", "submit_summary", {"summary": "done"})], "tool_use"
            )
        )
        run_anthropic_task(task, client=client)
        assert fake.messages.calls[0]["model"] == "claude-opus-4-7"

    def test_model_can_be_overridden(self) -> None:
        task = find_and_synthesise(2)
        client, fake = _client(
            _Response(
                [_tool_block("c1", "submit_summary", {"summary": "done"})], "tool_use"
            )
        )
        run_anthropic_task(task, client=client, model="claude-sonnet-4-6")
        assert fake.messages.calls[0]["model"] == "claude-sonnet-4-6"

    def test_thinking_is_off_by_default(self) -> None:
        task = find_and_synthesise(2)
        client, fake = _client(
            _Response(
                [_tool_block("c1", "submit_summary", {"summary": "done"})], "tool_use"
            )
        )
        run_anthropic_task(task, client=client)
        assert "thinking" not in fake.messages.calls[0]

    def test_thinking_can_be_enabled(self) -> None:
        task = find_and_synthesise(2)
        client, fake = _client(
            _Response(
                [_tool_block("c1", "submit_summary", {"summary": "done"})], "tool_use"
            )
        )
        run_anthropic_task(task, client=client, thinking={"type": "adaptive"})
        assert fake.messages.calls[0]["thinking"] == {"type": "adaptive"}

    def test_parallel_tool_use_is_allowed_by_default(self) -> None:
        task = find_and_synthesise(2)
        client, fake = _client(
            _Response(
                [_tool_block("c1", "submit_summary", {"summary": "done"})], "tool_use"
            )
        )
        run_anthropic_task(task, client=client)
        assert "tool_choice" not in fake.messages.calls[0]

    def test_disable_parallel_tool_use_sets_tool_choice(self) -> None:
        task = find_and_synthesise(2)
        client, fake = _client(
            _Response(
                [_tool_block("c1", "submit_summary", {"summary": "done"})], "tool_use"
            )
        )
        run_anthropic_task(task, client=client, disable_parallel_tool_use=True)
        assert fake.messages.calls[0]["tool_choice"] == {
            "type": "auto",
            "disable_parallel_tool_use": True,
        }


class TestToolResultRoundTrip:
    """The adapter sends tool results back to the API in the correct shape."""

    def test_tool_result_is_forwarded_to_the_next_api_call(self) -> None:
        task = find_and_synthesise(2)
        client, fake = _client(
            _Response([_tool_block("c1", "lookup", {"topic": "topic_1"})], "tool_use"),
            _Response([_tool_block("c2", "submit_summary", {"summary": "ok"})], "tool_use"),
        )
        run_anthropic_task(task, client=client)
        second_call_messages = fake.messages.calls[1]["messages"]
        last_message = second_call_messages[-1]
        assert last_message["role"] == "user"
        block = last_message["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "c1"
        assert not block["is_error"]

    def test_tool_errors_are_marked_in_trajectory_and_api(self) -> None:
        task = find_and_synthesise(2)
        client, fake = _client(
            _Response([_tool_block("c1", "lookup", {"topic": "nope"})], "tool_use"),
            _Response([_tool_block("c2", "submit_summary", {"summary": "ok"})], "tool_use"),
        )
        trajectory = run_anthropic_task(task, client=client)
        tool_turns = [t for t in trajectory.turns if t.role == "tool"]
        assert tool_turns[0].tool_results[0].is_error
        second_call = fake.messages.calls[1]
        assert second_call["messages"][-1]["content"][0]["is_error"]


class TestUsageTracking:
    """The adapter accumulates per-call usage into trajectory metadata."""

    def test_usage_aggregates_across_iterations(self) -> None:
        task = find_and_synthesise(2)
        client, _ = _client(
            _Response(
                [_tool_block("c1", "lookup", {"topic": "topic_1"})],
                "tool_use",
                usage=_usage(input_tokens=100, output_tokens=10),
            ),
            _Response(
                [_tool_block("c2", "submit_summary", {"summary": "ok"})],
                "tool_use",
                usage=_usage(input_tokens=120, output_tokens=15),
            ),
        )
        trajectory = run_anthropic_task(task, client=client)
        usage = trajectory.metadata["usage"]
        assert usage["input_tokens"] == 220
        assert usage["output_tokens"] == 25
        assert usage["n_requests"] == 2


class TestAnnotationsFlow:
    """Probe annotations from the task land on the right trajectory turns."""

    def test_plant_annotation_lands_on_first_tool_turn(self) -> None:
        task = find_and_synthesise(2)
        client, _ = _client(
            _Response([_tool_block("c1", "lookup", {"topic": "topic_1"})], "tool_use"),
            _Response([_tool_block("c2", "submit_summary", {"summary": "ok"})], "tool_use"),
        )
        trajectory = run_anthropic_task(task, client=client)
        tool_turns = [t for t in trajectory.turns if t.role == "tool"]
        annotation = tool_turns[0].metadata.get("state_amnesia")
        assert annotation is not None
        assert annotation["role"] == "plant"
