"""Tests for the find_max adversarial task."""

from __future__ import annotations

import pytest

from halftrace import find_max, premature_termination
from halftrace.trajectory import Trajectory


class TestFindMaxConstruction:
    def test_requires_n_at_least_two(self) -> None:
        with pytest.raises(ValueError, match="n >= 2"):
            find_max(1)

    def test_requires_value_range_low_below_high(self) -> None:
        with pytest.raises(ValueError, match="low < high"):
            find_max(5, value_range=(10, 10))

    def test_seed_changes_values(self) -> None:
        t1 = find_max(5, seed=0)
        t2 = find_max(5, seed=1)
        assert t1._values != t2._values  # type: ignore[attr-defined]

    def test_instruction_decay_rule_in_metadata_when_enabled(self) -> None:
        task = find_max(5, instruction_decay_rule=True)
        assert "instruction_decay" in task.trajectory_metadata
        rule = task.trajectory_metadata["instruction_decay"]
        assert rule["rule_id"] == "starts_with_pattern"

    def test_no_instruction_decay_metadata_when_disabled(self) -> None:
        task = find_max(5, instruction_decay_rule=False)
        assert "instruction_decay" not in task.trajectory_metadata

    def test_tool_specs_include_lookup_and_submit(self) -> None:
        task = find_max(5)
        names = [t.name for t in task.tool_specs]
        assert "lookup" in names
        assert "submit" in names


class TestFindMaxToolCalls:
    def test_lookup_returns_value(self) -> None:
        task = find_max(5, seed=42)
        response = task.handle_tool_call("lookup", {"topic": "topic_1"})
        assert not response.is_error
        assert "value of topic_1 is" in response.result

    def test_lookup_unknown_topic_errors(self) -> None:
        task = find_max(5, seed=42)
        response = task.handle_tool_call("lookup", {"topic": "topic_999"})
        assert response.is_error

    def test_submit_terminates(self) -> None:
        task = find_max(5, seed=42)
        assert not task.is_done()
        task.handle_tool_call("submit", {"value": 500})
        assert task.is_done()

    def test_submit_annotation_includes_completion_data(self) -> None:
        task = find_max(5, seed=42)
        # Look up 3 of 5 topics, then submit.
        task.handle_tool_call("lookup", {"topic": "topic_1"})
        task.handle_tool_call("lookup", {"topic": "topic_2"})
        task.handle_tool_call("lookup", {"topic": "topic_3"})
        response = task.handle_tool_call("submit", {"value": 500})
        annotations = response.annotations
        assert annotations["premature_termination"]["n_required"] == 5
        assert annotations["premature_termination"]["n_completed"] == 3

    def test_duplicate_lookups_do_not_increase_completion_count(self) -> None:
        task = find_max(5, seed=42)
        task.handle_tool_call("lookup", {"topic": "topic_1"})
        task.handle_tool_call("lookup", {"topic": "topic_1"})  # duplicate
        task.handle_tool_call("lookup", {"topic": "topic_2"})
        response = task.handle_tool_call("submit", {"value": 0})
        assert response.annotations["premature_termination"]["n_completed"] == 2


class TestFindMaxScoring:
    def test_premature_termination_picks_up_turn_annotation(self) -> None:
        task = find_max(10, seed=0)
        # Simulate a trajectory where the agent looks up 4 of 10 topics
        # and submits.
        trajectory = Trajectory()
        trajectory.add_turn("system", content=task.system_prompt)
        for i in range(4):
            task.handle_tool_call("lookup", {"topic": f"topic_{i + 1}"})
        submit_response = task.handle_tool_call("submit", {"value": 0})
        trajectory.add_turn(
            "tool",
            metadata=submit_response.annotations,
        )
        score = premature_termination(trajectory)
        assert score.value == 0.4
        assert score.details == {"n_required": 10, "n_completed": 4}
