"""Cold-start cure replication on the marker rule (rule v2: front-loaded [[END]]).

The content-rule replication (mechanism_content_rule.py) found that Sonnet
4.6 scores 10/10 on the 'include the word investigating' rule at baseline
— no failure to cure, so the cure couldn't be tested. The marker rule
from earlier pilot phase (front-loaded `[[END]]`) is the right replication
target: earlier pilots showed Sonnet bimodal-abandons on this rule at
N=10 (per-rep scores 0.091, 0.000, 0.818), so it has the baseline failure
that the cure is meant to fix.

If the four primer conditions produce the same dose-response (baseline
bimodal, zero tri-modal, minimal mostly-bimodal, full perfect) as on the
STATUS counter rule, the cold-start dynamic is rule-family-general on
Sonnet. If they don't, the cure is specific to counter/state-tracking
rules.

10 reps at N=10 across 4 conditions = 40 trajectories. ~$1.50.
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
MARKER = "[[END]]"

MARKER_RULE_PROMPT_HEADER = (
    "You are a research assistant.\n\n"
    f"OUTPUT FORMAT (must be followed every single turn): end every text "
    f"response you produce with the literal marker `{MARKER}` on its own. "
    "The marker must be the very last token of your reply.\n\n"
    "Example of a correctly formatted assistant response:\n"
    f"  Looking up topic_3 now. {MARKER}\n\n"
)


def override_task_with_marker_rule(task: Any) -> None:
    """Mutate find_and_synthesise to use the end-with-marker rule.

    Replaces the system prompt header with the marker rule and swaps the
    `instruction_decay` annotation to `end_with_marker`. The task body
    (lookup instructions, tool calls) is preserved.
    """
    base_prompt = task.system_prompt
    task_section_start = base_prompt.find("TASK:")
    if task_section_start == -1:
        raise RuntimeError("Could not find 'TASK:' header in task system_prompt")
    task_section = base_prompt[task_section_start:]
    task.system_prompt = MARKER_RULE_PROMPT_HEADER + task_section
    task.trajectory_metadata["instruction_decay"] = {
        "rule_id": "end_with_marker",
        "params": {"marker": MARKER},
    }


CONDITIONS: dict[str, list[dict[str, Any]]] = {
    "baseline": [],
    "zero": [
        {"role": "user", "content": "Hi."},
    ],
    "minimal": [
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": "ok."},
    ],
    "full": [
        {"role": "user", "content": "Quick warm-up before the real task: what is 2 + 2?"},
        {"role": "assistant", "content": "2 + 2 is 4."},
        {"role": "user", "content": "Another warm-up: what is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
    ],
}


def run_condition(
    cond_name: str, primer: list[dict[str, Any]], client: anthropic.Anthropic
) -> None:
    out_path = Path(f"results/mechanism_marker/{cond_name}_{MODEL}.jsonl")
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
        print(f"[{cond_name}] all reps done", file=sys.stderr)
        return
    print(
        f"[{cond_name}] running {len(plan)} reps on {MODEL} (marker [[END]] rule)",
        file=sys.stderr,
        flush=True,
    )
    for rep in plan:
        print(f"  + {cond_name} N=10 rep={rep}", file=sys.stderr, flush=True)
        task = find_and_synthesise(10, seed=rep, n_plants=1, discovery=False)
        override_task_with_marker_rule(task)
        trajectory = run_with_primer(
            task, primer=primer, model=MODEL, client=client
        )
        scores = score_all(trajectory)
        record: dict[str, Any] = {
            "n": 10,
            "rep": rep,
            "model": MODEL,
            "rule": f"end_with_marker/{MARKER}",
            "condition": cond_name,
            "scores": {name: s.model_dump(mode="json") for name, s in scores.items()},
            "trajectory": trajectory.model_dump(mode="json"),
        }
        with out_path.open("a") as f:
            f.write(json.dumps(record) + "\n")


def main() -> int:
    client = anthropic.Anthropic(max_retries=20, timeout=180.0)
    for cond_name, primer in CONDITIONS.items():
        run_condition(cond_name, primer, client)
    print("all conditions done", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
