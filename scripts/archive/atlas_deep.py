"""Deep atlas pilot: sweep five frontier models at higher N values.

The shallow atlas (`scripts/atlas_pilot.py`) ran N in {5, 10, 25} and
showed everything-perfect except a bimodal Sonnet cell that vanished by
N=25. This script runs N in {50, 100, 200} on the same five models with
5 reps each to test whether decay or other shapes appear at properly
long trajectory lengths.

Differences from atlas_pilot.py:

- Constructs adapter-specific clients with very generous retry budgets
  (max_retries=30) so transient rate-limits don't lose data.
- Skips gpt-4o at N=200 by default — at that depth, accumulated context
  per request exceeds OpenAI's 30K TPM ceiling on the standard tier and
  the cell can't complete in reasonable time. Pass `--include-gpt4o-n200`
  if you have a higher tier.
- Appends results to results/atlas_deep/<model>.jsonl so reruns are
  additive rather than destructive.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from halftrace.tasks import find_and_synthesise

MODELS: list[str] = [
    "claude-sonnet-4-6",
    "claude-opus-4-7",
    "claude-haiku-4-5",
    "gpt-4.1",
    "gpt-4o",
]
N_VALUES: list[int] = [50, 100, 200]
REPS: int = 5

PROBES = {
    "state_amnesia": state_amnesia,
    "instruction_decay": instruction_decay,
    "tool_repetition": tool_repetition,
    "narration_substitution": narration_substitution,
}


def _make_client(model: str) -> object:
    if model.startswith("claude-"):
        return anthropic.Anthropic(max_retries=20, timeout=180.0)
    if model.startswith("gpt-"):
        return openai.OpenAI(max_retries=30, timeout=300.0)
    raise ValueError(f"unknown provider for {model!r}")


def _run_trial(model: str, client: object, n: int, rep: int):
    task = find_and_synthesise(n, seed=rep, n_plants=1, discovery=False)
    if model.startswith("claude-"):
        trajectory = run_anthropic_task(
            task,
            model=model,
            max_tokens=4096,
            max_iterations=500,
            client=client,  # type: ignore[arg-type]
            disable_parallel_tool_use=True,
        )
    else:
        trajectory = run_openai_task(
            task,
            model=model,
            max_tokens=4096,
            max_iterations=500,
            client=client,  # type: ignore[arg-type]
            disable_parallel_tool_use=True,
        )
    scores = {name: probe(trajectory) for name, probe in PROBES.items()}
    return trajectory, scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-gpt4o-n200",
        action="store_true",
        help="Include gpt-4o at N=200 (typically fails on standard 30K TPM tier).",
    )
    args = parser.parse_args(argv)

    output_dir = Path("results/atlas_deep")
    output_dir.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        print(f"\n=== {model} ===", file=sys.stderr, flush=True)
        out_path = output_dir / f"{model}.jsonl"
        client = _make_client(model)

        count = 0
        for n in N_VALUES:
            if model == "gpt-4o" and n == 200 and not args.include_gpt4o_n200:
                print("  [skip] gpt-4o at N=200 (TPM ceiling)", file=sys.stderr)
                continue
            for rep in range(REPS):
                count += 1
                print(f"[{count}] N={n} rep={rep}", file=sys.stderr, flush=True)
                trajectory, scores = _run_trial(model, client, n, rep)
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

        # crude per-model usage summary
        try:
            with out_path.open() as f:
                lines = f.readlines()
            in_tokens = 0
            out_tokens = 0
            for line in lines[-count:] if count > 0 else []:
                r = json.loads(line)
                usage = r["trajectory"]["metadata"].get("usage", {})
                in_tokens += usage.get("input_tokens", 0)
                out_tokens += usage.get("output_tokens", 0)
            print(
                f"  trajectories={count}  in={in_tokens:,}  out={out_tokens:,}",
                file=sys.stderr,
            )
        except Exception as exc:
            print(f"  (usage summary skipped: {exc})", file=sys.stderr)

    print("\n=== Deep atlas complete ===", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
