"""Just gpt-4o deep at N=50, 100 (skips N=200 — TPM ceiling).

Brings gpt-4o to parity with the other models in the deep atlas. 5 reps
each, resume-aware.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import openai

from halftrace.adapters.openai_adapter import run_openai_task
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
    out_path = Path("results/atlas_deep/gpt-4o.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[tuple[int, int]] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            done.add((int(r["n"]), int(r["rep"])))

    plan = [
        (n, rep) for n in [50, 100] for rep in range(5) if (n, rep) not in done
    ]
    if not plan:
        print("all gpt-4o deep cells done", file=sys.stderr)
        return 0

    print(f"running {len(plan)} gpt-4o deep cells", file=sys.stderr, flush=True)
    client = openai.OpenAI(max_retries=30, timeout=300.0)

    for n, rep in plan:
        print(f"  + gpt-4o N={n} rep={rep}", file=sys.stderr, flush=True)
        task = find_and_synthesise(n, seed=rep, n_plants=1, discovery=False)
        trajectory = run_openai_task(
            task,
            model="gpt-4o",
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
            "model": "gpt-4o",
            "scores": {name: s.model_dump(mode="json") for name, s in scores.items()},
            "trajectory": trajectory.model_dump(mode="json"),
        }
        with out_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    print("done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
