"""Pilot the find_max adversarial task to exercise premature_termination.

find_and_synthesise enumerates the work upfront, so tool_repetition,
narration_substitution, and premature_termination all scored 1.0
everywhere on every model and N tested. find_max keeps the same lookup
mechanic but removes the "look up all N" instruction and gives the agent
explicit license to be efficient. Agents that take the bait submit
before exhausting lookups, which the probe scores as a partial
n_completed / n_required ratio.

Runs 10 reps at N=10 on Sonnet 4.6 by default. Pass --models to compare
multiple models on the same task.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from halftrace.probes import (
    Score,
    instruction_decay,
    narration_substitution,
    premature_termination,
    state_amnesia,
    tool_repetition,
)
from halftrace.runner import _select_adapter, estimate_cost
from halftrace.tasks import find_max
from halftrace.trajectory import Trajectory

PROBES: dict[str, Any] = {
    "state_amnesia": state_amnesia,
    "instruction_decay": instruction_decay,
    "tool_repetition": tool_repetition,
    "narration_substitution": narration_substitution,
    "premature_termination": premature_termination,
}


def run_one(model: str, n: int, rep: int) -> tuple[Trajectory, Mapping[str, Score]]:
    adapter = _select_adapter(model)
    task = find_max(n, seed=rep, instruction_decay_rule=True)
    trajectory = adapter(
        task,
        model=model,
        max_tokens=4096,
        max_iterations=500,
        disable_parallel_tool_use=True,
    )
    scores = {name: probe(trajectory) for name, probe in PROBES.items()}
    return trajectory, scores


def run_model(model: str, n: int, reps: int, out_dir: Path) -> None:
    out_path = out_dir / f"{model.replace('/', '_')}.jsonl"

    done: set[int] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            done.add(int(r["rep"]))

    plan = [rep for rep in range(reps) if rep not in done]
    if not plan:
        print(f"[{model}] all reps done", file=sys.stderr)
        return

    print(f"[{model}] running {len(plan)} reps at N={n}", file=sys.stderr, flush=True)
    usage_totals: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }

    with out_path.open("a") as f:
        for rep in plan:
            print(f"  + {model} N={n} rep={rep}", file=sys.stderr, flush=True)
            trajectory, scores = run_one(model, n, rep)
            u = trajectory.metadata.get("usage", {}) or trajectory.metadata.get(
                "usage_totals", {}
            )
            for k in usage_totals:
                usage_totals[k] += int(u.get(k, 0))

            find_max_meta: dict[str, Any] = {}
            for turn in trajectory.turns:
                fm = turn.metadata.get("find_max")
                if fm:
                    find_max_meta = fm
                    break

            record: dict[str, Any] = {
                "n": n,
                "rep": rep,
                "model": model,
                "scores": {name: s.model_dump(mode="json") for name, s in scores.items()},
                "find_max": find_max_meta,
                "trajectory": trajectory.model_dump(mode="json"),
            }
            f.write(json.dumps(record) + "\n")

    cost = estimate_cost(model, usage_totals)
    if cost is not None:
        print(f"[{model}] approx cost ${cost:.3f}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["claude-sonnet-4-6"])
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(f"results/find_max/n={args.n}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        run_model(model, args.n, args.reps, out_dir)

    print("done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
