"""Sonnet 4.6 at intermediate N values (35, 70) — characterise the decay shape.

The shallow atlas shows perfect compliance at N=25, the deep atlas shows
~10% commit at N=100. The transition happens somewhere in between.
This script fills in N=35 and N=70 with 10 reps each, giving us a
sharp picture of where the bimodal coin flips.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import anthropic

from halftrace.adapters.anthropic_adapter import run_anthropic_task
from halftrace.probes import (
    instruction_decay,
    narration_substitution,
    state_amnesia,
    tool_repetition,
)
from halftrace.probes.base import Score
from halftrace.tasks import find_and_synthesise

PROBES = {
    "state_amnesia": state_amnesia,
    "instruction_decay": instruction_decay,
    "tool_repetition": tool_repetition,
    "narration_substitution": narration_substitution,
}


def main() -> int:
    out_path = Path("results/atlas_intermediate/claude-sonnet-4-6.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[tuple[int, int]] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            done.add((int(r["n"]), int(r["rep"])))

    plan = [
        (n, rep)
        for n in [35, 70]
        for rep in range(10)
        if (n, rep) not in done
    ]
    if not plan:
        print("all sonnet intermediate cells done", file=sys.stderr)
        return 0

    print(f"running {len(plan)} sonnet intermediate cells", file=sys.stderr, flush=True)
    client = anthropic.Anthropic(max_retries=20, timeout=180.0)

    for n, rep in plan:
        print(f"  + sonnet N={n} rep={rep}", file=sys.stderr, flush=True)
        task = find_and_synthesise(n, seed=rep, n_plants=1, discovery=False)
        trajectory = run_anthropic_task(
            task,
            model="claude-sonnet-4-6",
            max_tokens=4096,
            max_iterations=500,
            client=client,
            disable_parallel_tool_use=True,
        )
        scores: Mapping[str, Score] = {
            name: probe(trajectory) for name, probe in PROBES.items()
        }
        record: dict[str, Any] = {
            "n": n,
            "rep": rep,
            "model": "claude-sonnet-4-6",
            "scores": {name: s.model_dump(mode="json") for name, s in scores.items()},
            "trajectory": trajectory.model_dump(mode="json"),
        }
        with out_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    print("done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
