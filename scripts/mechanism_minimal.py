"""Minimal-primer control: how degenerate can the prior turn be?

The mechanism test (scripts/mechanism_test.py) and its no-format control
(scripts/mechanism_control.py) both drove Sonnet 4.6's N=10 instruction
compliance to 10/10 perfect, against a baseline of 1/5 perfect. That
falsifies the strong "agent commits based on seeing the format in prior
turns" hypothesis: two off-topic exchanges (2+2, capital of Paris) with
plain-prose assistant responses already close the gap.

This script tightens the screw. The primer is the smallest possible
prior exchange: one user message, one trivial assistant reply ("ok.").
If 10/10 still holds, the warm-up effect is about presence-of-prior-turn,
not coherence-of-prior-content. If compliance regresses toward the
0.345 baseline, two exchanges of substance matter and the one-turn
primer isn't enough.
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
CONDITION = "minimal_one_turn_primer"
# The smallest possible prior exchange. One user turn, one trivial
# assistant reply, no format, no content of substance.
PRIMER_MESSAGES: list[dict[str, Any]] = [
    {"role": "user", "content": "Hi."},
    {"role": "assistant", "content": "ok."},
]


def main() -> int:
    out_path = Path("results/mechanism_minimal") / f"{MODEL}.jsonl"
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
        print("all mechanism minimal cells done", file=sys.stderr)
        return 0

    print(
        f"running {len(plan)} MINIMAL N=10 sonnet trajectories (one-turn primer)",
        file=sys.stderr,
        flush=True,
    )
    client = anthropic.Anthropic(max_retries=20, timeout=180.0)

    for rep in plan:
        print(f"  + minimal N=10 rep={rep}", file=sys.stderr, flush=True)
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
