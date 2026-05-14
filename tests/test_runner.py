"""Tests for the pilot runner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from halftrace import Score, Trajectory
from halftrace.runner import (
    estimate_cost,
    parse_args,
    run_pilot,
)


def _fake_trial(value: float, usage: dict[str, int] | None = None):
    def trial(n: int, rep: int) -> tuple[Trajectory, Score]:
        t = Trajectory(task_id=f"fake/n={n}/rep={rep}")
        if usage is not None:
            t.metadata["usage"] = usage
        s = Score(probe="state_amnesia", value=value, n_observations=1)
        return t, s

    return trial


def _decaying_trial(n_to_value: dict[int, float]):
    def trial(n: int, rep: int) -> tuple[Trajectory, Score]:
        t = Trajectory(task_id=f"fake/n={n}/rep={rep}")
        s = Score(probe="state_amnesia", value=n_to_value[n], n_observations=1)
        return t, s

    return trial


class TestEstimateCost:
    """Token totals translate to USD using the known pricing table."""

    def test_known_model_returns_a_dollar_amount(self) -> None:
        cost = estimate_cost(
            "claude-sonnet-4-6",
            {
                "input_tokens": 1_000_000,
                "output_tokens": 1_000_000,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        )
        assert cost is not None
        assert abs(cost - (3.0 + 15.0)) < 1e-9

    def test_unknown_model_returns_none(self) -> None:
        assert estimate_cost("not-a-model", {"input_tokens": 100}) is None

    def test_cache_reads_are_cheaper_than_uncached_input(self) -> None:
        no_cache = estimate_cost(
            "claude-sonnet-4-6", {"input_tokens": 1_000_000}
        )
        all_cache = estimate_cost(
            "claude-sonnet-4-6", {"cache_read_input_tokens": 1_000_000}
        )
        assert no_cache is not None and all_cache is not None
        assert all_cache < no_cache

    def test_cache_writes_cost_more_than_uncached_input(self) -> None:
        no_cache = estimate_cost(
            "claude-sonnet-4-6", {"input_tokens": 1_000_000}
        )
        all_writes = estimate_cost(
            "claude-sonnet-4-6", {"cache_creation_input_tokens": 1_000_000}
        )
        assert no_cache is not None and all_writes is not None
        assert all_writes > no_cache


class TestRunPilotValidation:
    """Input validation at the public boundary."""

    def test_empty_n_values_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="at least one N"):
            run_pilot(
                model="x",
                n_values=[],
                reps=1,
                output_path=tmp_path / "out.jsonl",
                trial=_fake_trial(1.0),
            )

    def test_zero_reps_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="reps >= 1"):
            run_pilot(
                model="x",
                n_values=[5],
                reps=0,
                output_path=tmp_path / "out.jsonl",
                trial=_fake_trial(1.0),
            )


class TestRunPilotOutput:
    """JSONL output shape and content."""

    def test_writes_one_record_per_trial(self, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        result = run_pilot(
            model="claude-sonnet-4-6",
            n_values=[5, 10],
            reps=2,
            output_path=out,
            trial=_fake_trial(1.0),
        )
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 4
        assert result.n_trajectories == 4

    def test_each_record_contains_score_and_trajectory(self, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        run_pilot(
            model="claude-sonnet-4-6",
            n_values=[5],
            reps=1,
            output_path=out,
            trial=_fake_trial(0.5),
        )
        record = json.loads(out.read_text().strip())
        assert record["n"] == 5
        assert record["rep"] == 0
        assert record["model"] == "claude-sonnet-4-6"
        assert record["score"]["value"] == 0.5
        assert "trajectory" in record

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        out = tmp_path / "nested" / "subdir" / "out.jsonl"
        run_pilot(
            model="x",
            n_values=[5],
            reps=1,
            output_path=out,
            trial=_fake_trial(1.0),
        )
        assert out.exists()


class TestHalftraceFitting:
    """The runner fits a halftrace from collected scores."""

    def test_clean_decay_curve_produces_a_halftrace(self, tmp_path: Path) -> None:
        result = run_pilot(
            model="x",
            n_values=[5, 10, 20],
            reps=3,
            output_path=tmp_path / "out.jsonl",
            trial=_decaying_trial({5: 1.0, 10: 0.7, 20: 0.2}),
        )
        assert result.halftrace is not None
        assert result.halftrace.value is not None
        assert 10 < result.halftrace.value < 20

    def test_no_crossing_yields_none_value(self, tmp_path: Path) -> None:
        result = run_pilot(
            model="x",
            n_values=[5, 10],
            reps=2,
            output_path=tmp_path / "out.jsonl",
            trial=_fake_trial(0.9),
        )
        assert result.halftrace is not None
        assert result.halftrace.value is None


class TestUsageAggregation:
    """The runner sums per-trajectory usage across the sweep."""

    def test_token_totals_aggregate_across_trials(self, tmp_path: Path) -> None:
        usage = {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 10,
            "cache_read_input_tokens": 200,
        }
        result = run_pilot(
            model="claude-sonnet-4-6",
            n_values=[5],
            reps=3,
            output_path=tmp_path / "out.jsonl",
            trial=_fake_trial(1.0, usage=usage),
        )
        assert result.total_input_tokens == 300
        assert result.total_output_tokens == 150
        assert result.total_cache_creation_tokens == 30
        assert result.total_cache_read_tokens == 600

    def test_estimated_cost_is_computed_for_known_models(
        self, tmp_path: Path
    ) -> None:
        usage = {
            "input_tokens": 100_000,
            "output_tokens": 10_000,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
        result = run_pilot(
            model="claude-sonnet-4-6",
            n_values=[5],
            reps=1,
            output_path=tmp_path / "out.jsonl",
            trial=_fake_trial(1.0, usage=usage),
        )
        assert result.estimated_cost_usd is not None
        assert abs(result.estimated_cost_usd - 0.45) < 1e-9

    def test_trajectories_without_usage_are_handled(self, tmp_path: Path) -> None:
        result = run_pilot(
            model="claude-sonnet-4-6",
            n_values=[5],
            reps=1,
            output_path=tmp_path / "out.jsonl",
            trial=_fake_trial(1.0),
        )
        assert result.total_input_tokens == 0


class TestProgress:
    """Optional progress callback is invoked per trial."""

    def test_progress_callback_fires_once_per_trial(self, tmp_path: Path) -> None:
        messages: list[str] = []
        run_pilot(
            model="x",
            n_values=[5, 10],
            reps=2,
            output_path=tmp_path / "out.jsonl",
            trial=_fake_trial(1.0),
            progress=messages.append,
        )
        assert len(messages) == 4


class TestArgParse:
    """CLI argument parsing."""

    def test_default_arguments(self) -> None:
        args = parse_args([])
        assert args.model == "claude-sonnet-4-6"
        assert args.n == [5, 10, 25]
        assert args.reps == 3
        assert args.output == Path("results/pilot.jsonl")
        assert not args.dry_run

    def test_overrides_are_applied(self) -> None:
        args = parse_args(
            [
                "--model",
                "claude-opus-4-7",
                "--n",
                "5",
                "10",
                "--reps",
                "5",
                "--dry-run",
            ]
        )
        assert args.model == "claude-opus-4-7"
        assert args.n == [5, 10]
        assert args.reps == 5
        assert args.dry_run
