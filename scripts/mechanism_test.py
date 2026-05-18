"""Test the early-uncertainty mechanism hypothesis at Sonnet N=10.

Baseline N=10 compliance for Sonnet on the STATUS counter rule is 0.58 — about
4 in 10 trajectories abandon the rule after turn 4 and never pick it up.

The hypothesis in the post:

    The agent's commitment to a format rule depends on whether it has seen
    the rule applied in its own previous text turns. At N=5 there's only one
    text turn before the first tool result. At N=10 the agent has produced
    a couple of turns, but not enough to have firmly resolved "is this format
    real or vestigial." By N=25 it has seen itself apply the rule 5+ times,
    and the prior shifts to "yes, this is part of how I write."

Test: inject two prior user-assistant exchanges where the assistant has
already produced compliant STATUS: 1 / STATUS: 2 responses, then let the
real task run as N=10. If the hypothesis is right, compliance should jump
from ~0.58 to ~0.95+. If it doesn't move, the hypothesis is wrong.

The primer turns are NOT counted in the scored trajectory — only the
real-task assistant turns are scored. This keeps the comparison clean
against baseline N=10.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import anthropic
from _primer_runner import run_with_primer, score_all

from halftrace.tasks import find_and_synthesise

MODEL = "claude-sonnet-4-6"
CONDITION = "primed_2_examples"
PRIMER_MESSAGES: list[dict[str, Any]] = [
    {"role": "user", "content": "Quick warm-up before the real task: what is 2 + 2?"},
    {"role": "assistant", "content": "STATUS: 1\n2 + 2 is 4."},
    {"role": "user", "content": "Another warm-up: what is the capital of France?"},
    {"role": "assistant", "content": "STATUS: 2\nThe capital of France is Paris."},
]


def main() -> int:
    out_path = Path("results/mechanism_test") / f"{MODEL}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[int] = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            done.add(int(r["rep"]))

    plan = [rep for rep in range(10) if rep not in done]
    if not plan:
        print("all mechanism test cells done", file=sys.stderr)
        return 0

    print(
        f"running {len(plan)} primed N=10 sonnet trajectories",
        file=sys.stderr,
        flush=True,
    )
    client = anthropic.Anthropic(max_retries=20, timeout=180.0)

    for rep in plan:
        print(f"  + primed N=10 rep={rep}", file=sys.stderr, flush=True)
        task = find_and_synthesise(10, seed=rep, n_plants=1, discovery=False)
        trajectory = run_with_primer(
            task, primer=PRIMER_MESSAGES, model=MODEL, client=client
        )
        scores = score_all(trajectory)
        record: dict[str, Any] = {
            "n": 10,
            "rep": rep,
            "model": MODEL,
            "condition": CONDITION,
            "scores": {name: s.model_dump(mode="json") for name, s in scores.items()},
            "trajectory": trajectory.model_dump(mode="json"),
        }
        with out_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    print("done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
