"""Tests for the Trajectory schema."""

from __future__ import annotations

import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from halftrace import ToolCall, ToolResult, Trajectory, Turn


class TestTrajectoryConstruction:
    """Building trajectories turn-by-turn."""

    def test_empty_trajectory_has_sensible_defaults(self) -> None:
        t = Trajectory()
        assert t.turns == []
        assert t.task_id is None
        assert t.model is None
        assert t.metadata == {}
        assert isinstance(t.id, str) and len(t.id) > 0

    def test_each_trajectory_gets_a_unique_id(self) -> None:
        t1 = Trajectory()
        t2 = Trajectory()
        assert t1.id != t2.id

    def test_add_turn_returns_the_created_turn(self) -> None:
        t = Trajectory()
        turn = t.add_turn(role="user", content="hello")
        assert turn.role == "user"
        assert turn.content == "hello"
        assert turn.index == 0

    def test_turn_indices_increment_in_insertion_order(self) -> None:
        t = Trajectory()
        t.add_turn(role="user", content="first")
        t.add_turn(role="assistant", content="second")
        t.add_turn(role="user", content="third")
        assert [turn.index for turn in t.turns] == [0, 1, 2]

    def test_turn_with_only_role_has_none_content(self) -> None:
        t = Trajectory()
        turn = t.add_turn(role="assistant")
        assert turn.content is None
        assert turn.tool_calls == []
        assert turn.tool_results == []

    def test_metadata_is_attached_per_turn(self) -> None:
        t = Trajectory()
        turn = t.add_turn(role="user", content="hello", metadata={"source": "test"})
        assert turn.metadata == {"source": "test"}


class TestToolCallsAndResults:
    """Tool call attachment and the flattened views."""

    def test_tool_call_attaches_to_assistant_turn(self) -> None:
        t = Trajectory()
        tc = ToolCall(id="t1", name="read_file", args={"path": "/a.txt"})
        turn = t.add_turn(role="assistant", tool_calls=[tc])
        assert len(turn.tool_calls) == 1
        assert turn.tool_calls[0].name == "read_file"

    def test_tool_result_attaches_to_tool_turn(self) -> None:
        t = Trajectory()
        tr = ToolResult(tool_call_id="t1", name="read_file", result="contents")
        turn = t.add_turn(role="tool", tool_results=[tr])
        assert len(turn.tool_results) == 1
        assert turn.tool_results[0].result == "contents"

    def test_flattened_tool_calls_returns_index_and_call(self) -> None:
        t = Trajectory()
        t.add_turn(role="user", content="do thing")
        t.add_turn(
            role="assistant",
            tool_calls=[
                ToolCall(id="t1", name="read_file", args={"path": "/a.txt"}),
                ToolCall(id="t2", name="read_file", args={"path": "/b.txt"}),
            ],
        )
        t.add_turn(
            role="tool",
            tool_results=[
                ToolResult(tool_call_id="t1", name="read_file", result="A"),
                ToolResult(tool_call_id="t2", name="read_file", result="B"),
            ],
        )

        calls = t.tool_calls()
        assert len(calls) == 2
        assert calls[0] == (1, t.turns[1].tool_calls[0])
        assert calls[1] == (1, t.turns[1].tool_calls[1])

    def test_flattened_tool_results_returns_index_and_result(self) -> None:
        t = Trajectory()
        t.add_turn(role="user", content="do thing")
        t.add_turn(
            role="assistant",
            tool_calls=[ToolCall(id="t1", name="read_file", args={})],
        )
        t.add_turn(
            role="tool",
            tool_results=[ToolResult(tool_call_id="t1", name="read_file", result="A")],
        )

        results = t.tool_results()
        assert len(results) == 1
        assert results[0][0] == 2
        assert results[0][1].result == "A"

    def test_tool_result_can_be_an_error(self) -> None:
        tr = ToolResult(tool_call_id="t1", name="read_file", result="ENOENT", is_error=True)
        assert tr.is_error is True

    def test_tool_call_args_accept_arbitrary_json(self) -> None:
        tc = ToolCall(
            id="t1",
            name="complex",
            args={"nested": {"deep": [1, 2, {"x": "y"}]}, "n": 5},
        )
        assert tc.args["nested"]["deep"][2]["x"] == "y"


class TestSerialisation:
    """JSONL round-trip and serialisation guarantees."""

    def test_jsonl_round_trip_is_lossless(self) -> None:
        original = Trajectory(task_id="task_42", model="claude-sonnet-4-6")
        original.add_turn(role="user", content="hi")
        original.add_turn(
            role="assistant",
            tool_calls=[ToolCall(id="t1", name="read_file", args={"path": "/a"})],
        )
        original.add_turn(
            role="tool",
            tool_results=[ToolResult(tool_call_id="t1", name="read_file", result="A")],
        )

        line = original.to_jsonl_line()
        restored = Trajectory.from_jsonl_line(line)

        assert restored == original

    def test_jsonl_line_contains_no_newlines(self) -> None:
        t = Trajectory()
        t.add_turn(role="user", content="line one\nline two\nline three")
        line = t.to_jsonl_line()
        assert "\n" not in line

    def test_jsonl_line_is_valid_json(self) -> None:
        t = Trajectory()
        t.add_turn(role="user", content="hello")
        line = t.to_jsonl_line()
        parsed = json.loads(line)
        assert parsed["turns"][0]["role"] == "user"

    def test_timestamp_serialises_as_iso_utc(self) -> None:
        t = Trajectory()
        t.add_turn(role="user", content="hi")
        parsed = json.loads(t.to_jsonl_line())
        timestamp = parsed["turns"][0]["timestamp"]
        # ISO-8601 with timezone info
        parsed_dt = datetime.fromisoformat(timestamp)
        offset = parsed_dt.utcoffset()
        assert offset is not None
        assert offset.total_seconds() == 0


class TestValidation:
    """Schema validation: things that should fail loudly."""

    def test_unknown_role_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Turn(index=0, role="banana")  # type: ignore[arg-type]

    def test_extra_field_on_trajectory_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Trajectory.model_validate({"turns": [], "mystery_field": "x"})

    def test_extra_field_on_turn_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Turn.model_validate({"index": 0, "role": "user", "content": "hi", "extra": "x"})

    def test_tool_call_requires_id_name_args(self) -> None:
        with pytest.raises(ValidationError):
            ToolCall.model_validate({"name": "read_file"})  # missing id and args


class TestTokenEstimate:
    """The chars/4 token estimate."""

    def test_empty_trajectory_has_zero_tokens(self) -> None:
        t = Trajectory()
        assert t.token_count_estimate() == 0

    def test_estimate_includes_content(self) -> None:
        t = Trajectory()
        t.add_turn(role="user", content="a" * 40)  # 40 chars
        assert t.token_count_estimate() == 10  # 40 / 4

    def test_estimate_includes_tool_calls_and_results(self) -> None:
        t = Trajectory()
        t.add_turn(
            role="assistant",
            tool_calls=[ToolCall(id="t1", name="x", args={"k": "v"})],
        )
        t.add_turn(
            role="tool",
            tool_results=[ToolResult(tool_call_id="t1", name="x", result="r" * 20)],
        )
        # The exact value isn't load-bearing; just confirm it grows.
        assert t.token_count_estimate() > 0


class TestImmutabilityPattern:
    """Trajectories are conceptually append-only.

    Pydantic doesn't enforce this, but we test the intended usage pattern
    so that any regression is noisy.
    """

    def test_add_turn_only_appends(self) -> None:
        t = Trajectory()
        t.add_turn(role="user", content="first")
        t.add_turn(role="assistant", content="second")
        assert len(t.turns) == 2
        # Indices should be assignment-order, not insertion-order tricks
        assert t.turns[0].index == 0
        assert t.turns[1].index == 1
