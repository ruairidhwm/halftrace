"""Pilot runner: sweep (N, rep) cells, score each trajectory, profile compliance.

Run a pilot on Sonnet 4.6 across small N values:

    uv run halftrace --n 5 10 25 --reps 3

The runner writes one JSON record per trajectory to `--output` (default
`results/pilot.jsonl`) and prints token totals, an estimated cost, and a
per-probe compliance profile (shape + commit probability, with a halftrace
only when the shape is `gradient`) to stderr. The Anthropic adapter is
imported lazily so `--help` and `--dry-run` work without the optional extra.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

from halftrace.compare import ProbeComparison, compare_profiles
from halftrace.diagnose import diagnose
from halftrace.fit import ComplianceProfile, analyse_compliance
from halftrace.probes import (
    Score,
    instruction_decay,
    narration_substitution,
    state_amnesia,
    tool_repetition,
)
from halftrace.probes.base import Probe
from halftrace.tasks import find_and_synthesise
from halftrace.trajectory import Trajectory

TrialFn = Callable[[int, int], tuple[Trajectory, dict[str, Score]]]
ProgressFn = Callable[[str], None]

PROBES: dict[str, Probe] = {
    "state_amnesia": state_amnesia,
    "instruction_decay": instruction_decay,
    "tool_repetition": tool_repetition,
    "narration_substitution": narration_substitution,
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
    profiles: dict[str, ComplianceProfile | None] = {}
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

    profiles: dict[str, ComplianceProfile | None] = {}
    for probe_name, by_n in scores_by_probe_and_n.items():
        if by_n:
            profiles[probe_name] = analyse_compliance(by_n, probe=probe_name)
        else:
            profiles[probe_name] = None

    return PilotResult(
        model_name=model,
        n_values=n_values,
        reps=reps,
        n_trajectories=n_trajectories,
        profiles=profiles,
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
    n_plants: int,
    discovery: bool,
) -> TrialFn:
    from halftrace.adapters import run_anthropic_task

    def trial(n: int, rep: int) -> tuple[Trajectory, dict[str, Score]]:
        task = find_and_synthesise(
            n, seed=rep, n_plants=n_plants, discovery=discovery
        )
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
        description=(
            "Halftrace: diagnose how your agent's compliance with rules varies "
            "across trajectory length. Two subcommands: `pilot` (run new "
            "trajectories against an LLM) and `analyse` (ingest existing logs)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pilot = sub.add_parser(
        "pilot",
        help="Run a new pilot sweep against the Anthropic API.",
    )
    pilot.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Anthropic model ID (default: claude-sonnet-4-6).",
    )
    pilot.add_argument(
        "--n",
        nargs="+",
        type=int,
        default=[5, 10, 25],
        help="N values to sweep (default: 5 10 25).",
    )
    pilot.add_argument(
        "--reps",
        type=int,
        default=3,
        help="Repetitions per N (default: 3).",
    )
    pilot.add_argument(
        "--output",
        type=Path,
        default=Path("results/pilot.jsonl"),
        help="JSONL output path (default: results/pilot.jsonl).",
    )
    pilot.add_argument("--max-tokens", type=int, default=4096)
    pilot.add_argument("--max-iterations", type=int, default=500)
    pilot.add_argument(
        "--n-plants",
        type=int,
        default=1,
        help=(
            "Number of state_amnesia plants per trajectory (default: 1). "
            "Must be <= min(N) - 1 across all --n values."
        ),
    )
    pilot.add_argument(
        "--serial",
        action="store_true",
        help=(
            "Disable parallel tool use: force one tool call per assistant turn. "
            "Required to exercise context decay over N tool-call rounds."
        ),
    )
    pilot.add_argument(
        "--discovery",
        action="store_true",
        help=(
            "Use adversarial discovery mode: agent does not get the topic list "
            "upfront and must maintain its own 'seen' state via discover_next."
        ),
    )
    pilot.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exit without making API calls.",
    )

    analyse = sub.add_parser(
        "analyse",
        help="Analyse existing trajectory logs without hitting any API.",
    )
    analyse.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Path to a JSONL file. Each line is one message payload in the "
            "format selected by --format."
        ),
    )
    analyse.add_argument(
        "--format",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="Input message format (default: anthropic).",
    )
    analyse.add_argument(
        "--commit-threshold",
        type=float,
        default=0.95,
        help=(
            "Score at which a trajectory counts as having committed to a "
            "rule (default: 0.95). Used to compute commit_probability."
        ),
    )

    compare = sub.add_parser(
        "compare",
        help=(
            "Compare two sets of trajectory logs (e.g. before / after a "
            "prompt change) and report per-probe deltas."
        ),
    )
    compare.add_argument(
        "--before", type=Path, required=True, help="JSONL path of baseline logs."
    )
    compare.add_argument(
        "--after", type=Path, required=True, help="JSONL path of new logs."
    )
    compare.add_argument(
        "--format",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="Input message format (default: anthropic).",
    )
    compare.add_argument(
        "--commit-threshold",
        type=float,
        default=0.95,
        help=(
            "Score at which a trajectory counts as having committed to a "
            "rule (default: 0.95). Used to compute commit_probability."
        ),
    )
    compare.add_argument(
        "--delta-threshold",
        type=float,
        default=0.05,
        help=(
            "Minimum commit_probability change to classify as improved "
            "or regressed (default: 0.05)."
        ),
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "pilot":
        return _run_pilot_command(args)
    if args.command == "analyse":
        return _run_analyse_command(args)
    if args.command == "compare":
        return _run_compare_command(args)
    raise AssertionError(f"unknown command {args.command!r}")  # argparse should catch


def _run_pilot_command(args: argparse.Namespace) -> int:
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
        args.model,
        args.max_tokens,
        args.max_iterations,
        serial=args.serial,
        n_plants=args.n_plants,
        discovery=args.discovery,
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
    for probe_name, profile in result.profiles.items():
        if profile is None:
            print(
                f"{probe_name}: no observations (no annotations in any trajectory)",
                file=sys.stderr,
            )
            continue
        _print_profile_with_diagnosis(probe_name, profile)

    return 0


def _print_profile_with_diagnosis(probe_name: str, profile: ComplianceProfile) -> None:
    line = (
        f"{probe_name}: shape={profile.shape}  "
        f"commit_p={profile.commit_probability:.2f}"
    )
    if profile.shape == "gradient" and profile.halftrace is not None:
        line += f"  halftrace={profile.halftrace:.2f}"
    elif profile.shape == "gradient":
        line += "  halftrace=undefined (no crossing in range)"
    print(line, file=sys.stderr)
    if profile.shape == "perfect":
        return
    diagnosis = diagnose(profile)
    print(f"  why:  {diagnosis.cause}", file=sys.stderr)
    print("  try:", file=sys.stderr)
    for suggestion in diagnosis.suggestions:
        print(f"    - {suggestion}", file=sys.stderr)


def _run_analyse_command(args: argparse.Namespace) -> int:
    trajectories = _load_trajectories(args.input, args.format)
    print(
        f"Analysing {len(trajectories)} trajectories from {args.input} "
        f"({args.format} format)",
        file=sys.stderr,
    )

    profiles = _profile_trajectories(trajectories, args.commit_threshold)
    print("", file=sys.stderr)
    for probe_name in PROBES:
        profile = profiles.get(probe_name)
        if profile is None:
            print(
                f"{probe_name}: no observations (no annotations in any trajectory)",
                file=sys.stderr,
            )
            continue
        _print_profile_with_diagnosis(probe_name, profile)

    return 0


def _run_compare_command(args: argparse.Namespace) -> int:
    before_traj = _load_trajectories(args.before, args.format)
    after_traj = _load_trajectories(args.after, args.format)
    print(
        f"Comparing {len(before_traj)} before-trajectories vs "
        f"{len(after_traj)} after-trajectories ({args.format} format)",
        file=sys.stderr,
    )

    before_profiles = _profile_trajectories(before_traj, args.commit_threshold)
    after_profiles = _profile_trajectories(after_traj, args.commit_threshold)
    comparisons = compare_profiles(
        before_profiles,
        after_profiles,
        delta_threshold=args.delta_threshold,
    )

    print("", file=sys.stderr)
    for cmp in comparisons:
        _print_comparison(cmp)

    return 0


def _load_trajectories(path: Path, fmt: str) -> list[Trajectory]:
    from halftrace.ingest import from_anthropic_messages, from_openai_messages

    parser_fn = from_anthropic_messages if fmt == "anthropic" else from_openai_messages
    trajectories: list[Trajectory] = []
    with path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            payload = cast("dict[str, Any]", json.loads(line))
            trajectories.append(parser_fn(payload))
    return trajectories


def _profile_trajectories(
    trajectories: list[Trajectory],
    commit_threshold: float,
) -> dict[str, ComplianceProfile | None]:
    """Score every probe across the given trajectories and emit a profile per probe."""
    scores_by_probe: dict[str, list[float]] = {name: [] for name in PROBES}
    for trajectory in trajectories:
        for probe_name, probe in PROBES.items():
            try:
                score = probe(trajectory)
            except ValueError:
                continue
            if score.value is not None:
                scores_by_probe[probe_name].append(score.value)

    profiles: dict[str, ComplianceProfile | None] = {}
    for probe_name in PROBES:
        scores = scores_by_probe[probe_name]
        if not scores:
            profiles[probe_name] = None
            continue
        profiles[probe_name] = analyse_compliance(
            {0: scores},
            probe=probe_name,
            commit_threshold=commit_threshold,
        )
    return profiles


def _print_comparison(cmp: ProbeComparison) -> None:
    icon = {
        "improved": "[+]",
        "regressed": "[-]",
        "unchanged": "[ ]",
        "appeared": "[+]",
        "disappeared": "[-]",
        "missing": "[ ]",
    }[cmp.direction]

    def fmt(p: ComplianceProfile | None) -> str:
        if p is None:
            return "none"
        return f"shape={p.shape} c={p.commit_probability:.2f}"

    delta_str = ""
    if cmp.commit_probability_delta is not None:
        delta_str = f"  Δcommit={cmp.commit_probability_delta:+.2f}"
    shape_str = ""
    if cmp.shape_changed and cmp.before is not None and cmp.after is not None:
        shape_str = f"  shape: {cmp.before.shape} → {cmp.after.shape}"
    line = (
        f"{icon} {cmp.probe}: {fmt(cmp.before)} → {fmt(cmp.after)}  "
        f"({cmp.direction}){delta_str}{shape_str}"
    )
    print(line, file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
