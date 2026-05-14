"""Pilot runner: sweep (N, rep) cells, score each trajectory, fit a halftrace.

Run a state_amnesia pilot on Sonnet 4.6 across small N values:

    uv run halftrace --n 5 10 25 --reps 3

The runner writes one JSON record per trajectory to `--output` (default
`results/pilot.jsonl`) and prints token totals, an estimated cost, and the
fitted halftrace + 95% CI to stderr. The Anthropic adapter is imported
lazily so `--help` and `--dry-run` work without the optional extra.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

from halftrace.fit import Halftrace, fit_halftrace
from halftrace.probes import Score, instruction_decay, state_amnesia
from halftrace.probes.base import Probe
from halftrace.tasks import find_and_synthesise
from halftrace.trajectory import Trajectory

TrialFn = Callable[[int, int], tuple[Trajectory, dict[str, Score]]]
ProgressFn = Callable[[str], None]

PROBES: dict[str, Probe] = {
    "state_amnesia": state_amnesia,
    "instruction_decay": instruction_decay,
}

_PRICING_PER_M_TOKENS: dict[str, tuple[float, float]] = {
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


class PilotResult(BaseModel):
    """Aggregate result of a pilot sweep."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_name: str
    n_values: list[int]
    reps: int
    n_trajectories: int
    halftraces: dict[str, Halftrace | None] = {}
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    estimated_cost_usd: float | None = None


def estimate_cost(model: str, usage: dict[str, int]) -> float | None:
    """Estimate USD cost from accumulated usage counters.

    Returns None for unknown models. Cache writes are billed at 1.25x input
    price, cache reads at 0.1x.
    """
    if model not in _PRICING_PER_M_TOKENS:
        return None
    in_price, out_price = _PRICING_PER_M_TOKENS[model]
    return (
        (usage.get("input_tokens", 0) / 1_000_000) * in_price
        + (usage.get("cache_creation_input_tokens", 0) / 1_000_000) * in_price * 1.25
        + (usage.get("cache_read_input_tokens", 0) / 1_000_000) * in_price * 0.1
        + (usage.get("output_tokens", 0) / 1_000_000) * out_price
    )


def run_pilot(
    *,
    model: str,
    n_values: list[int],
    reps: int,
    output_path: Path,
    trial: TrialFn,
    progress: ProgressFn | None = None,
) -> PilotResult:
    """Run a pilot sweep and write trajectory records to `output_path`.

    `trial(n, rep)` runs one cell and returns `(trajectory, score)`. The
    runner writes one JSONL record per trial, accumulates token totals
    across trials from `trajectory.metadata["usage"]`, and fits a halftrace
    from non-None score values once all trials are complete.
    """
    if not n_values:
        raise ValueError("run_pilot requires at least one N value")
    if reps < 1:
        raise ValueError(f"run_pilot requires reps >= 1, got {reps}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    scores_by_probe_and_n: dict[str, dict[int, list[float]]] = {}
    totals: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    n_trajectories = 0

    with output_path.open("w") as f:
        for n in n_values:
            for rep in range(reps):
                n_trajectories += 1
                if progress is not None:
                    progress(f"[{n_trajectories}] N={n} rep={rep}")

                trajectory, scores = trial(n, rep)

                for probe_name, score in scores.items():
                    by_n = scores_by_probe_and_n.setdefault(probe_name, {})
                    if score.value is not None:
                        by_n.setdefault(n, []).append(score.value)

                raw_usage = trajectory.metadata.get("usage")
                if isinstance(raw_usage, dict):
                    usage = cast("dict[str, Any]", raw_usage)
                    for key in totals:
                        value = usage.get(key, 0)
                        if isinstance(value, int):
                            totals[key] += value

                record = {
                    "n": n,
                    "rep": rep,
                    "model": model,
                    "scores": {
                        name: s.model_dump(mode="json") for name, s in scores.items()
                    },
                    "trajectory": trajectory.model_dump(mode="json"),
                }
                f.write(json.dumps(record) + "\n")
                f.flush()

    halftraces: dict[str, Halftrace | None] = {}
    for probe_name, by_n in scores_by_probe_and_n.items():
        if len(by_n) >= 2:
            halftraces[probe_name] = fit_halftrace(by_n, n_bootstrap=1000)
        else:
            halftraces[probe_name] = None

    return PilotResult(
        model_name=model,
        n_values=n_values,
        reps=reps,
        n_trajectories=n_trajectories,
        halftraces=halftraces,
        total_input_tokens=totals["input_tokens"],
        total_output_tokens=totals["output_tokens"],
        total_cache_read_tokens=totals["cache_read_input_tokens"],
        total_cache_creation_tokens=totals["cache_creation_input_tokens"],
        estimated_cost_usd=estimate_cost(model, totals),
    )


def _default_trial(
    model: str,
    max_tokens: int,
    max_iterations: int,
    *,
    serial: bool,
) -> TrialFn:
    from halftrace.adapters import run_anthropic_task

    def trial(n: int, rep: int) -> tuple[Trajectory, dict[str, Score]]:
        task = find_and_synthesise(n, seed=rep)
        trajectory = run_anthropic_task(
            task,
            model=model,
            max_tokens=max_tokens,
            max_iterations=max_iterations,
            disable_parallel_tool_use=serial,
        )
        scores = {name: probe(trajectory) for name, probe in PROBES.items()}
        return trajectory, scores

    return trial


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="halftrace",
        description="Run a state_amnesia pilot across N values and fit a halftrace.",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Anthropic model ID (default: claude-sonnet-4-6).",
    )
    parser.add_argument(
        "--n",
        nargs="+",
        type=int,
        default=[5, 10, 25],
        help="N values to sweep (default: 5 10 25).",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=3,
        help="Repetitions per N (default: 3).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pilot.jsonl"),
        help="JSONL output path (default: results/pilot.jsonl).",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument(
        "--serial",
        action="store_true",
        help=(
            "Disable parallel tool use: force one tool call per assistant turn. "
            "Required to exercise context decay over N tool-call rounds."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exit without making API calls.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    plan = [(n, rep) for n in args.n for rep in range(args.reps)]
    print(
        f"Plan: {len(plan)} trajectories ({len(args.n)} N values x {args.reps} reps)",
        file=sys.stderr,
    )
    print(f"Model:  {args.model}", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)

    if args.dry_run:
        for n, rep in plan:
            print(f"  N={n} rep={rep}", file=sys.stderr)
        return 0

    trial = _default_trial(
        args.model, args.max_tokens, args.max_iterations, serial=args.serial
    )
    result = run_pilot(
        model=args.model,
        n_values=args.n,
        reps=args.reps,
        output_path=args.output,
        trial=trial,
        progress=lambda msg: print(msg, file=sys.stderr, flush=True),
    )

    print("\n=== Pilot results ===", file=sys.stderr)
    print(f"Trajectories:           {result.n_trajectories}", file=sys.stderr)
    print(f"Input tokens:           {result.total_input_tokens:,}", file=sys.stderr)
    print(f"Output tokens:          {result.total_output_tokens:,}", file=sys.stderr)
    print(f"Cache read tokens:      {result.total_cache_read_tokens:,}", file=sys.stderr)
    print(f"Cache creation tokens:  {result.total_cache_creation_tokens:,}", file=sys.stderr)
    if result.estimated_cost_usd is not None:
        print(f"Estimated cost:         ${result.estimated_cost_usd:.4f}", file=sys.stderr)

    print("", file=sys.stderr)
    for probe_name, h in result.halftraces.items():
        if h is None:
            print(
                f"{probe_name}: no data (no annotations or fewer than 2 N values)",
                file=sys.stderr,
            )
            continue
        if h.value is None:
            print(
                f"{probe_name}: no crossing of {h.threshold} in tested N range",
                file=sys.stderr,
            )
        else:
            line = f"{probe_name}: halftrace = {h.value:.2f}"
            if h.ci_low is not None and h.ci_high is not None:
                line += f"  95% CI [{h.ci_low:.2f}, {h.ci_high:.2f}]"
            line += (
                f"  (bootstrap {h.n_bootstrap_resolved}/{h.n_bootstrap} resolved)"
            )
            print(line, file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
