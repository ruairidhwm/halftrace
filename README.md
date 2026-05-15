# halftrace

**Diagnose how your agent's compliance with rules varies across trajectory length.**

[![CI](https://github.com/ruairidhwm/halftrace/actions/workflows/test.yml/badge.svg)](https://github.com/ruairidhwm/halftrace/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/halftrace.svg)](https://pypi.org/project/halftrace/)
[![Python](https://img.shields.io/pypi/pyversions/halftrace.svg)](https://pypi.org/project/halftrace/)
[![License](https://img.shields.io/pypi/l/halftrace.svg)](LICENSE)

`halftrace` is a diagnostic instrument for agent trajectories. Point it at your existing OpenAI or Anthropic message logs and it tells you, per failure mode:

- **what shape** of compliance your agent has (perfect / abandoned / bimodal / categorical / gradient),
- **why** that shape is likely showing up, and
- **what to try next** based on patterns we observed in twelve API pilots.

No new API calls required. No new task harness to wire up. You bring the logs; `halftrace` does the analysis.

## What the empirical record says

Halftrace was originally designed to measure *gradual decay* of agent capabilities over long trajectories — a concept inherited from the "context rot" intuition. Twelve pilot runs (~64 trajectories, $8.77 of spend, see [`RESULTS.md`](RESULTS.md)) produced a different picture:

> **Modern Claude (Sonnet 4.6, Haiku 4.5) does not decay gradually on simple agentic tasks at N up to 200. Compliance is *categorical per turn-type* and *bimodal per trajectory* — the agent either commits to a rule for the whole trajectory or abandons it from turn 4 onward, with the choice approximately a coinflip.**

Three rule designs, two task variants, two models, N from 5 to 200, zero instances of gradient decay on any probe. The original `halftrace` scalar (the trajectory length at which compliance crosses 50%) is undefined for the shapes we actually observe, so `halftrace` now reports the shape of the compliance pattern and a `commit_probability` (fraction of trajectories that follow the rule end-to-end) as the headline metrics. Halftrace-the-scalar is still reported when the shape is `gradient` — it just doesn't fire on the data we have.

## Three workflows

### Analyse existing logs

Most agent developers already have trajectory logs. Point `halftrace` at them:

```bash
halftrace analyse --input my_logs.jsonl --format openai
```

Each line of `my_logs.jsonl` is one trajectory as an OpenAI chat-completions payload (or Anthropic `messages.create()` payload, with `--format anthropic`). The tool ingests each, scores every probe, and prints a profile per probe:

```
Analysing 124 trajectories from my_logs.jsonl (openai format)

state_amnesia: shape=perfect  commit_p=1.00
instruction_decay: shape=bimodal  commit_p=0.52
  why:  Agent commits-or-abandons per trajectory: the choice is approximately
        a coinflip and is stable for the rest of the trajectory once made...
  try:
    - Add a worked example response in the system prompt...
    - Restate the rule in the initial user message...
    - Consider relaxing the rule on turn 1...
tool_repetition: shape=perfect  commit_p=1.00
narration_substitution: shape=perfect  commit_p=1.00
```

### Compare before and after a prompt change

The iterative prompt-engineering workflow:

```bash
halftrace compare --before before_logs.jsonl --after after_logs.jsonl --format openai
```

```
Comparing 50 before-trajectories vs 50 after-trajectories (openai format)

[+] instruction_decay: shape=bimodal c=0.52 → shape=perfect c=0.96  (improved)  Δcommit=+0.44  shape: bimodal → perfect
[ ] narration_substitution: shape=perfect c=1.00 → shape=perfect c=1.00  (unchanged)
[ ] state_amnesia: shape=perfect c=1.00 → shape=perfect c=1.00  (unchanged)
[ ] tool_repetition: shape=perfect c=1.00 → shape=perfect c=1.00  (unchanged)
```

No API spend; the comparison runs entirely over your existing logs.

### Run new trajectories against the Anthropic API

If you want a controlled experiment rather than working from production logs:

```bash
halftrace pilot --n 5 10 25 --reps 3 --serial
```

This drives a built-in synthetic task (`find_and_synthesise`) through Claude at varying trajectory lengths, scores every probe, and emits a profile. See `halftrace pilot --help` for the model, plant-count, and discovery-mode flags. `ANTHROPIC_API_KEY` must be set.

## What `halftrace` measures

Four probes ship in the box. Each scores a different failure mode per trajectory:

| Probe | What it measures |
| --- | --- |
| `state_amnesia` | Retention of facts planted earlier in the trajectory |
| `instruction_decay` | Adherence to a system-prompt rule over time |
| `tool_repetition` | Avoidance of re-calling tools with identical arguments |
| `narration_substitution` | Emitting tool calls rather than just describing them |

The pilot phase also drafted `premature_termination` (declaring the task done before all expected work) — it's not yet implemented.

For each probe, halftrace classifies the *shape* of the per-trajectory score distribution:

| Shape | When it fires | Headline metric |
| --- | --- | --- |
| `perfect` | All trajectories score ≥ 0.95 | `commit_probability` |
| `abandoned` | All trajectories score ≤ 0.05 | `commit_probability` |
| `bimodal` | High within-cell variance — coinflip per trajectory | `commit_probability` |
| `categorical` | Stable intermediate compliance — agent applies the rule to one turn-type and drops it on another | `commit_probability` |
| `gradient` | Monotone decreasing means — the case the original halftrace concept assumed | `halftrace` (also commit_probability) |
| `unclassified` | None of the above; usually means more reps or wider N needed | `commit_probability` |

Each non-perfect shape comes with a one-line cause and 2–3 concrete suggestions drawn from the empirical patterns in [`RESULTS.md`](RESULTS.md).

## Custom probes

A probe is a function `Trajectory -> Score`. The four shipped probes are 50-100 line files; you can drop in your own:

```python
from halftrace import Score, Trajectory


def too_many_apologies(trajectory: Trajectory) -> Score:
    """Score the fraction of assistant turns that DON'T start with an apology."""
    text_turns = [
        t for t in trajectory.turns
        if t.role == "assistant" and t.content
    ]
    if not text_turns:
        return Score(probe="too_many_apologies", value=None, n_observations=0)
    bad = sum(
        1 for t in text_turns
        if t.content is not None and t.content.lower().startswith("i apologise")
    )
    return Score(
        probe="too_many_apologies",
        value=(len(text_turns) - bad) / len(text_turns),
        n_observations=len(text_turns),
    )
```

Then score it across your logs and analyse the shape:

```python
from halftrace import from_openai_messages, analyse_compliance, diagnose

trajectories = [from_openai_messages(payload) for payload in load_my_logs()]
scores = [too_many_apologies(t).value for t in trajectories]
scores = [s for s in scores if s is not None]

profile = analyse_compliance({0: scores}, probe="too_many_apologies")
print(f"{profile.shape}: {profile.commit_probability:.2f}")
print(diagnose(profile).cause)
```

Probes that need configuration (e.g. `instruction_decay`'s rule) read it from `Trajectory.metadata`. Tasks set this metadata; ingested trajectories receive it from the source payload's optional `metadata` field.

## What this isn't

- **Not a benchmark.** No leaderboard, no canonical task set. The shape classifier characterises *your* agent on *your* logs.
- **Not an eval framework.** If you want grading, scoring rubrics, or production observability, use [Inspect](https://inspect.ai-safety-institute.org.uk/), [Braintrust](https://www.braintrust.dev/), or [LangSmith](https://www.langchain.com/langsmith). `halftrace` is a diagnostic instrument that sits alongside them.
- **Not an agent framework.** It doesn't build agents. It diagnoses agents you've already built.

## Install

```bash
pip install halftrace
```

For the `pilot` subcommand against Claude:

```bash
pip install "halftrace[anthropic]"
```

Optional extras:

```bash
pip install "halftrace[openai]"  # not yet wired into the runner; reserved for future
pip install "halftrace[all]"
```

Requires Python 3.11+.

## Reading order

- [`README.md`](README.md) — this file: positioning, workflows, custom probes.
- [`RESULTS.md`](RESULTS.md) — pilot-phase findings, cost ledger, the "modern Claude commits or doesn't" claim with supporting data.
- [`HYPOTHESES.md`](HYPOTHESES.md) — the original pre-registered design (assumes gradient decay; preserved for transparency about what we expected vs. what we found).

## License

MIT.
