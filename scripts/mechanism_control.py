"""Control for the mechanism test: priming WITHOUT the format prefix.

The mechanism test (scripts/mechanism_test.py) showed that two priming turns
where the assistant has already produced compliant STATUS responses drives
Sonnet's N=10 compliance from 0.58 → 1.00.

That result is consistent with the hypothesis "agent commits based on seeing
the rule in its own previous turns." But it is also consistent with a
weaker hypothesis: "any prior assistant turns warm the agent up." If the
weaker version is true, primer turns that do NOT follow the format should
also close the dip.

This script runs the control. Same primer structure (two off-task user-
assistant exchanges before the real task) but the assistant's primer
responses have NO STATUS prefix.

Expected outcomes:

- If compliance stays around 0.58 (baseline): mechanism hypothesis
  confirmed. The format-in-prior-turns is what matters, not just priors.
- If compliance jumps to ~1.00: mechanism hypothesis falsified. The agent
  is just being warmed up by any prior context, regardless of format.
- Anything in between: partial effect; both stories are partly right.
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
CONDITION = "control_no_format_primer"
# Same primer Q&A as mechanism_test.py, but the assistant responses do NOT
# include the STATUS prefix. Tests whether the warm-up effect is
# specifically about seeing the format in prior turns or just about having
# any prior conversation.
PRIMER_MESSAGES: list[dict[str, Any]] = [
    {"role": "user", "content": "Quick warm-up before the real task: what is 2 + 2?"},
    {"role": "assistant", "content": "2 + 2 is 4."},
    {"role": "user", "content": "Another warm-up: what is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
]


def main() -> int:
    out_path = Path("results/mechanism_control") / f"{MODEL}.jsonl"
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
        print("all mechanism control cells done", file=sys.stderr)
        return 0

    print(
        f"running {len(plan)} CONTROL N=10 sonnet trajectories (no-format primer)",
        file=sys.stderr,
        flush=True,
    )
    client = anthropic.Anthropic(max_retries=20, timeout=180.0)

    for rep in plan:
        print(f"  + control N=10 rep={rep}", file=sys.stderr, flush=True)
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
