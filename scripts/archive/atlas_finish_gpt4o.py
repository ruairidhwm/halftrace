"""Finish the gpt-4o cell of the atlas pilot after a TPM rate limit.

The main atlas script (`scripts/atlas_pilot.py`) hit OpenAI's 30K TPM
rate limit on gpt-4o partway through the N=25 cell. This script runs
only the missing trajectories (N=25 reps 2, 3, 4) using a client with
a generous retry budget, and appends them to the existing jsonl.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import openai

from halftrace.adapters.openai_adapter import run_openai_task
from halftrace.probes import (
    instruction_decay,
    narration_substitution,
    state_amnesia,
    tool_repetition,
)
from halftrace.tasks import find_and_synthesise

MODEL = "gpt-4o"
OUTPUT_PATH = Path("results/atlas/gpt-4o.jsonl")
N = 25
REPS_TO_RUN = [2, 3, 4]

PROBES = {
    "state_amnesia": state_amnesia,
    "instruction_decay": instruction_decay,
    "tool_repetition": tool_repetition,
    "narration_substitution": narration_substitution,
}


def main() -> int:
    client = openai.OpenAI(max_retries=10, timeout=120.0)

    with OUTPUT_PATH.open("a") as f:
        for rep in REPS_TO_RUN:
            print(f"[finish] gpt-4o N={N} rep={rep}", file=sys.stderr, flush=True)
            task = find_and_synthesise(N, seed=rep, n_plants=1, discovery=False)
            trajectory = run_openai_task(
                task,
                model=MODEL,
                max_tokens=4096,
                max_iterations=500,
                client=client,
                disable_parallel_tool_use=True,
            )
            scores = {name: probe(trajectory) for name, probe in PROBES.items()}
            record = {
                "n": N,
                "rep": rep,
                "model": MODEL,
                "scores": {n: s.model_dump(mode="json") for n, s in scores.items()},
                "trajectory": trajectory.model_dump(mode="json"),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()

            # Brief pause between trajectories to stay under the TPM ceiling on
            # the next call's prompt-token count.
            time.sleep(30)

    print("done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
