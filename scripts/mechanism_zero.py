"""Zero-content primer: is the assistant turn the load-bearing piece?

The minimal-primer control (scripts/mechanism_minimal.py) showed a clean
dose-response on Sonnet 4.6 N=10 instruction compliance:

    baseline (no primer):                 0.345 mean, 1/5 perfect
    one trivial exchange (Hi. / ok.):     0.836 mean, 8/10 perfect
    two trivial exchanges (no format):    1.000 mean, 10/10 perfect

That establishes the effect is content-independent and quantitative.
Open question: is *any* prior turn enough, or specifically a prior
*assistant* turn? This script isolates that. The primer here is a single
extra user message before the real task's user message — no prior
assistant turn at all.

If compliance stays near baseline, the assistant turn is doing the
work: the agent commits more reliably when it has at least one prior
turn of *its own* in the visible context. If compliance jumps toward
1.0, what matters is just the position of the real task within the
conversation, not who spoke before.

Note: Anthropic's API accepts consecutive same-role messages without
merging them on the client side. The two `user` messages are sent as
distinct turns.
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
CONDITION = "zero_extra_user_only"
# One extra user turn before the real task's user message. No prior
# assistant turn anywhere in the visible context.
PRIMER_MESSAGES: list[dict[str, Any]] = [
    {"role": "user", "content": "Hi."},
]


def main() -> int:
    out_path = Path("results/mechanism_zero") / f"{MODEL}.jsonl"
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
        print("all mechanism zero cells done", file=sys.stderr)
        return 0

    print(
        f"running {len(plan)} ZERO N=10 sonnet trajectories "
        "(extra user msg, no assistant primer)",
        file=sys.stderr,
        flush=True,
    )
    client = anthropic.Anthropic(max_retries=20, timeout=180.0)

    for rep in plan:
        print(f"  + zero N=10 rep={rep}", file=sys.stderr, flush=True)
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
