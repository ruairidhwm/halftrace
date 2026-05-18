"""Scoped final pass: Sonnet intermediate N + GPT-4o deep at N=50, 100.

Skips the full 10-rep rigour pass that would double every cell. Instead
runs only the two pieces that materially strengthen the headline:

1. Sonnet 4.6 at N=35 and N=70, 10 reps each. Characterises where the
   decay transition between N=25 (perfect) and N=100 (~10% commit)
   actually happens.
2. GPT-4o at N=50 and N=100, 5 reps each. Completes the cross-provider
   deep grid (every other deep cell is at 5 reps; this brings GPT-4o
   to parity).

Resume-aware: skips cells already on disk.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import anthropic
import openai

from halftrace.adapters.anthropic_adapter import run_anthropic_task
from halftrace.adapters.openai_adapter import run_openai_task
from halftrace.probes import (
    instruction_decay,
    narration_substitution,
    state_amnesia,
    tool_repetition,
)
from halftrace.probes.base import Score
from halftrace.tasks import find_and_synthesise
from halftrace.trajectory import Trajectory

PROBES = {
    "state_amnesia": state_amnesia,
    "instruction_decay": instruction_decay,
    "tool_repetition": tool_repetition,
    "narration_substitution": narration_substitution,
}


def _load_existing(path: Path) -> set[tuple[int, int]]:
    if not path.exists():
        return set()
    done: set[tuple[int, int]] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        done.add((int(r["n"]), int(r["rep"])))
    return done


def _append(
    path: Path,
    n: int,
    rep: int,
    model: str,
    trajectory: Trajectory,
    scores: Mapping[str, Score],
) -> None:
    record: dict[str, Any] = {
        "n": n,
        "rep": rep,
        "model": model,
        "scores": {name: s.model_dump(mode="json") for name, s in scores.items()},
        "trajectory": trajectory.model_dump(mode="json"),
    }
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def main() -> int:
    # 1. Sonnet intermediate
    print("=== sonnet intermediate (N=35, 70) ===", file=sys.stderr, flush=True)
    sonnet_path = Path("results/atlas_intermediate/claude-sonnet-4-6.jsonl")
    sonnet_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_existing(sonnet_path)
    plan_sonnet = [
        (n, rep)
        for n in [35, 70]
        for rep in range(10)
        if (n, rep) not in done
    ]
    if plan_sonnet:
        client_a = anthropic.Anthropic(max_retries=20, timeout=180.0)
        for n, rep in plan_sonnet:
            print(f"  + sonnet N={n} rep={rep}", file=sys.stderr, flush=True)
            task = find_and_synthesise(n, seed=rep, n_plants=1, discovery=False)
            trajectory = run_anthropic_task(
                task,
                model="claude-sonnet-4-6",
                max_tokens=4096,
                max_iterations=500,
                client=client_a,
                disable_parallel_tool_use=True,
            )
            scores = {name: probe(trajectory) for name, probe in PROBES.items()}
            _append(sonnet_path, n, rep, "claude-sonnet-4-6", trajectory, scores)
    else:
        print("  (all sonnet intermediate cells already done)", file=sys.stderr)

    # 2. GPT-4o deep
    print("\n=== gpt-4o deep (N=50, 100) ===", file=sys.stderr, flush=True)
    gpt_path = Path("results/atlas_deep/gpt-4o.jsonl")
    gpt_path.parent.mkdir(parents=True, exist_ok=True)
    done_g = _load_existing(gpt_path)
    plan_gpt = [
        (n, rep)
        for n in [50, 100]
        for rep in range(5)
        if (n, rep) not in done_g
    ]
    if plan_gpt:
        client_o = openai.OpenAI(max_retries=30, timeout=300.0)
        for n, rep in plan_gpt:
            print(f"  + gpt-4o N={n} rep={rep}", file=sys.stderr, flush=True)
            task = find_and_synthesise(n, seed=rep, n_plants=1, discovery=False)
            trajectory = run_openai_task(
                task,
                model="gpt-4o",
                max_tokens=4096,
                max_iterations=500,
                client=client_o,
                disable_parallel_tool_use=True,
            )
            scores = {name: probe(trajectory) for name, probe in PROBES.items()}
            _append(gpt_path, n, rep, "gpt-4o", trajectory, scores)
    else:
        print("  (all gpt-4o deep cells already done)", file=sys.stderr)

    print("\n=== finish pass complete ===", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
