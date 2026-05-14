"""Tests for the find_and_synthesise task."""

from __future__ import annotations

import pytest

from halftrace import (
    FindAndSynthesise,
    ToolCall,
    ToolResult,
    Trajectory,
    find_and_synthesise,
    state_amnesia,
)


class TestConstruction:
    """Task construction and basic invariants."""

    def test_factory_and_class_produce_equivalent_tasks(self) -> None:
        t1 = find_and_synthesise(10, seed=0)
        t2 = FindAndSynthesise(10, seed=0)
        assert t1.id == t2.id
        assert t1.planted_codewords == t2.planted_codewords
        assert t1.topics == t2.topics

    def test_n_below_two_raises(self) -> None:
        with pytest.raises(ValueError, match="n >= 2"):
            find_and_synthesise(1)

    def test_n_plants_below_one_raises(self) -> None:
        with pytest.raises(ValueError, match="n_plants >= 1"):
            find_and_synthesise(10, n_plants=0)

    def test_n_plants_too_large_raises(self) -> None:
        with pytest.raises(ValueError, match="n_plants <= n - 1"):
            find_and_synthesise(5, n_plants=5)

    def test_id_encodes_n_k_and_seed(self) -> None:
        task = find_and_synthesise(25, n_plants=3, seed=7)
        assert "n=25" in task.id
        assert "k=3" in task.id
        assert "seed=7" in task.id

    def test_seed_makes_codewords_deterministic(self) -> None:
        a = find_and_synthesise(10, n_plants=3, seed=0)
        b = find_and_synthesise(10, n_plants=3, seed=0)
        assert a.planted_codewords == b.planted_codewords

    def test_different_seeds_produce_different_codewords(self) -> None:
        seen = {find_and_synthesise(10, seed=s).planted_codewords[0] for s in range(10)}
        assert len(seen) > 1

    def test_planted_codewords_length_matches_n_plants(self) -> None:
        task = find_and_synthesise(25, n_plants=5)
        assert len(task.planted_codewords) == 5
        # All codewords distinct (rng is seeded; for k=5 should hold)
        assert len(set(task.planted_codewords)) == 5

    def test_topics_match_n(self) -> None:
        task = find_and_synthesise(5)
        assert task.topics == [f"topic_{i + 1}" for i in range(5)]

    def test_tool_specs_expose_lookup_and_submit(self) -> None:
        task = find_and_synthesise(5)
        names = {spec.name for spec in task.tool_specs}
        assert names == {"lookup", "submit_summary"}


class TestLookupHandling:
    """The lookup tool: facts, plants, recalls, errors."""

    def test_first_lookup_returns_fact_and_plant_annotation(self) -> None:
        task = find_and_synthesise(5)
        response = task.handle_tool_call("lookup", {"topic": "topic_1"})
        assert not response.is_error
        assert task.planted_codewords[0] in response.result
        annotation = response.annotations["state_amnesia"]
        assert annotation["role"] == "plant"
        assert annotation["fact"] == task.planted_codewords[0]

    def test_middle_lookups_have_no_annotations(self) -> None:
        task = find_and_synthesise(5)
        task.handle_tool_call("lookup", {"topic": "topic_1"})  # plant
        response = task.handle_tool_call("lookup", {"topic": "topic_2"})
        assert response.annotations == {}
        assert not response.is_error

    def test_last_lookup_returns_fact_and_recall_annotation(self) -> None:
        task = find_and_synthesise(3)
        task.handle_tool_call("lookup", {"topic": "topic_1"})
        task.handle_tool_call("lookup", {"topic": "topic_2"})
        response = task.handle_tool_call("lookup", {"topic": "topic_3"})
        annotation = response.annotations["state_amnesia"]
        assert annotation["role"] == "recall"
        assert annotation["fact_ids"] == ["password_1"]
        assert "password" in response.result.lower()

    def test_plant_and_recall_share_fact_id(self) -> None:
        task = find_and_synthesise(2)
        first = task.handle_tool_call("lookup", {"topic": "topic_1"})
        last = task.handle_tool_call("lookup", {"topic": "topic_2"})
        assert first.annotations["state_amnesia"]["fact_id"] == "password_1"
        assert last.annotations["state_amnesia"]["fact_ids"] == ["password_1"]

    def test_multiple_plants_fire_at_evenly_spaced_positions(self) -> None:
        # n=25, k=5 → positions [0, 4, 9, 14, 19]; recall on lookup 24.
        task = find_and_synthesise(25, n_plants=5)
        plant_positions: list[int] = []
        for i, topic in enumerate(task.topics):
            response = task.handle_tool_call("lookup", {"topic": topic})
            ann = response.annotations.get("state_amnesia")
            if ann is not None and ann["role"] == "plant":
                plant_positions.append(i)
        assert plant_positions == [0, 4, 9, 14, 19]

    def test_recall_at_last_lookup_lists_all_fact_ids(self) -> None:
        task = find_and_synthesise(10, n_plants=3)
        responses = [
            task.handle_tool_call("lookup", {"topic": t}) for t in task.topics
        ]
        last_ann = responses[-1].annotations["state_amnesia"]
        assert last_ann["role"] == "recall"
        assert last_ann["fact_ids"] == ["password_1", "password_2", "password_3"]
        assert "3 passwords" in responses[-1].result

    def test_unknown_topic_returns_error(self) -> None:
        task = find_and_synthesise(5)
        response = task.handle_tool_call("lookup", {"topic": "not-a-topic"})
        assert response.is_error
        assert "unknown topic" in response.result.lower()

    def test_missing_topic_arg_returns_error(self) -> None:
        task = find_and_synthesise(5)
        response = task.handle_tool_call("lookup", {})
        assert response.is_error
        assert "topic" in response.result.lower()


class TestSubmitAndDone:
    """submit_summary ends the task; nothing else does."""

    def test_is_done_starts_false(self) -> None:
        assert not find_and_synthesise(5).is_done()

    def test_submit_summary_marks_task_done(self) -> None:
        task = find_and_synthesise(5)
        response = task.handle_tool_call("submit_summary", {"summary": "ok"})
        assert not response.is_error
        assert task.is_done()

    def test_submit_without_summary_arg_returns_error(self) -> None:
        task = find_and_synthesise(5)
        response = task.handle_tool_call("submit_summary", {})
        assert response.is_error
        assert not task.is_done()

    def test_lookups_do_not_mark_task_done(self) -> None:
        task = find_and_synthesise(2)
        task.handle_tool_call("lookup", {"topic": "topic_1"})
        task.handle_tool_call("lookup", {"topic": "topic_2"})
        assert not task.is_done()

    def test_unknown_tool_returns_error(self) -> None:
        task = find_and_synthesise(5)
        response = task.handle_tool_call("delete_everything", {})
        assert response.is_error


class TestEndToEndWithStateAmnesia:
    """A simulated transcript, scored by state_amnesia."""

    def _run_simulated(self, n: int, agent_recall_answer: str | None) -> Trajectory:
        """Drive a task without an LLM, building a trajectory by hand.

        `agent_recall_answer` is the assistant's text response after the
        final lookup; pass None to skip it (simulating an agent that goes
        straight to submit_summary).
        """
        task = find_and_synthesise(n)
        traj = Trajectory(task_id=task.id)
        traj.add_turn(role="system", content=task.system_prompt)
        traj.add_turn(role="user", content=task.initial_user_message)

        for i, topic in enumerate(task.topics):
            tc = ToolCall(id=f"c{i}", name="lookup", args={"topic": topic})
            traj.add_turn(role="assistant", tool_calls=[tc])
            response = task.handle_tool_call("lookup", {"topic": topic})
            traj.add_turn(
                role="tool",
                tool_results=[
                    ToolResult(tool_call_id=tc.id, name="lookup", result=response.result)
                ],
                metadata=response.annotations,
            )

        if agent_recall_answer is not None:
            traj.add_turn(role="assistant", content=agent_recall_answer)

        submit_tc = ToolCall(id="submit", name="submit_summary", args={"summary": "done"})
        traj.add_turn(role="assistant", tool_calls=[submit_tc])
        submit_response = task.handle_tool_call("submit_summary", {"summary": "done"})
        traj.add_turn(
            role="tool",
            tool_results=[
                ToolResult(
                    tool_call_id=submit_tc.id, name="submit_summary", result=submit_response.result
                )
            ],
        )
        return traj

    def test_agent_recalling_correctly_scores_one(self) -> None:
        # Both find_and_synthesise calls use seed=0 so the codewords match.
        task = find_and_synthesise(5)
        traj = self._run_simulated(
            5, agent_recall_answer=f"the password was {task.planted_codewords[0]}."
        )
        score = state_amnesia(traj)
        assert score.value == 1.0
        assert score.n_observations == 1

    def test_agent_failing_to_recall_scores_zero(self) -> None:
        traj = self._run_simulated(5, agent_recall_answer="I have no idea.")
        score = state_amnesia(traj)
        assert score.value == 0.0

    def test_agent_skipping_text_response_scores_zero(self) -> None:
        # No text turn between the final lookup and submit_summary.
        # state_amnesia looks at the next assistant turn with content;
        # the submit_summary turn has tool_calls but no text content,
        # so the probe sees no response and scores zero.
        traj = self._run_simulated(5, agent_recall_answer=None)
        score = state_amnesia(traj)
        assert score.value == 0.0
        assert score.details["per_fact"][0]["reason"] == "no_response"

    def test_n_two_still_produces_one_observation(self) -> None:
        # Smallest valid task: plant on lookup 0, recall on lookup 1.
        task = find_and_synthesise(2)
        traj = self._run_simulated(2, agent_recall_answer=task.planted_codewords[0])
        score = state_amnesia(traj)
        assert score.value == 1.0
        assert score.n_observations == 1
