"""Run only the GPT cells of the deep atlas.

After the main deep-atlas run exhausted the Anthropic credit balance
partway through Opus, this script finishes the OpenAI half (GPT-4.1
at N in {50, 100, 200} and GPT-4o at N in {50, 100}, skipping gpt-4o
at N=200 because of the 30K TPM ceiling).
"""

from __future__ import annotations

import json
import sys
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

REPS: int = 5
PROBES = {
    "state_amnesia": state_amnesia,
    "instruction_decay": instruction_decay,
    "tool_repetition": tool_repetition,
    "narration_substitution": narration_substitution,
}


def run_trial(client: openai.OpenAI, model: str, n: int, rep: int):
    task = find_and_synthesise(n, seed=rep, n_plants=1, discovery=False)
    trajectory = run_openai_task(
        task,
        model=model,
        max_tokens=4096,
        max_iterations=500,
        client=client,
        disable_parallel_tool_use=True,
    )
    scores = {name: probe(trajectory) for name, probe in PROBES.items()}
    return trajectory, scores


def main() -> int:
    output_dir = Path("results/atlas_deep")
    output_dir.mkdir(parents=True, exist_ok=True)
    client = openai.OpenAI(max_retries=30, timeout=300.0)

    plan: dict[str, list[int]] = {
        "gpt-4.1": [50, 100, 200],
        "gpt-4o": [50, 100],  # skip N=200 for TPM reasons
    }

    for model, n_values in plan.items():
        print(f"\n=== {model} ===", file=sys.stderr, flush=True)
        out_path = output_dir / f"{model}.jsonl"
        count = 0
        for n in n_values:
            for rep in range(REPS):
                count += 1
                print(f"[{count}] N={n} rep={rep}", file=sys.stderr, flush=True)
                trajectory, scores = run_trial(client, model, n, rep)
                record = {
                    "n": n,
                    "rep": rep,
                    "model": model,
                    "scores": {
                        name: s.model_dump(mode="json") for name, s in scores.items()
                    },
                    "trajectory": trajectory.model_dump(mode="json"),
                }
                with out_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")

    print("\n=== Deep GPT atlas complete ===", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
