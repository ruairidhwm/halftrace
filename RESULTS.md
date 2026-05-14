# Pilot results

Pre-pilot status, last updated 2026-05-14. The pilots described here are
scoped to validate the halftrace measurement pipeline against real model
behaviour at small N, before committing to the full 120-trajectory study
laid out in `HYPOTHESES.md`. They are not the full study.

## Summary

Eleven pilot runs against the Anthropic API (~60 trajectories, $8.26 of
spend) end-to-end validated the pipeline: `Task` → adapter → `Trajectory`
→ probe → `fit_halftrace` → CLI. All four `find_and_synthesise` probes
are implemented (`state_amnesia`, `instruction_decay`, `tool_repetition`,
`narration_substitution`) and have been measured against `claude-sonnet-4-6`
and `claude-haiku-4-5` across N values from 5 to 200, including multi-plant
state_amnesia variants and three different `instruction_decay` rule designs.

**No probe produced a halftrace in any configuration.** This is itself the
finding worth recording. The consistent pattern across the design space:

- `state_amnesia`, `tool_repetition`, `narration_substitution`: flat 1.0
  everywhere — the failure modes are not exercised by `find_and_synthesise`
  in the {5..200} range on modern Claude.
- `instruction_decay`: **bimodal coinflip per trajectory**, not smooth decay.
  The agent either commits to the rule for the whole trajectory or
  abandons it from turn 4 onward, with the choice approximately a coinflip.
  This held across three rule designs (buried marker, front-loaded marker,
  incrementing counter).

**Modern Claude on simple, well-specified agentic tasks does not decay
gradually — it commits or it doesn't.** The halftrace concept as defined
in `HYPOTHESES.md` assumes smooth degradation, which is the wrong shape
for the failure modes we observe.

## Pipeline status

| Component | Status |
| --- | --- |
| `Trajectory` schema + JSONL serialization | Done |
| `Probe` protocol + `Score` type | Done |
| `state_amnesia` probe (single- and multi-plant) | Done |
| `instruction_decay` probe (end_with_marker, starts_with_pattern) | Done |
| `tool_repetition` probe | Done |
| `narration_substitution` probe | Done |
| `premature_termination` probe | Not yet implemented |
| `fit_halftrace` (linear interp + bootstrap 95% CI) | Done |
| Anthropic adapter (manual tool-use loop, caching, usage tracking, `--serial`) | Done |
| OpenAI adapter | Not yet implemented |
| `find_and_synthesise` task (parameterised by `n_plants`) | Done |
| Pilot runner (`halftrace` CLI, multi-probe scoring, per-probe halftrace) | Done |

## Findings

### `state_amnesia`

| Variant | Model | N=5 | N=10 | N=25 | N=50 | N=100 | N=200 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| k=1 plant | `claude-sonnet-4-6` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| k=1 plant | `claude-haiku-4-5` | 1.000 | 1.000 | 1.000 | — | — | — |
| k=5 plants | `claude-sonnet-4-6` | — | 1.000 | 1.000 | — | 1.000 | — |

All cells scored 1.0; no decay observed. The single-plant probe correctly
captured a planted codeword across **400+ intervening trajectory turns**
at N=200 under forced serialization. The multi-plant probe (k=5 codewords
at evenly-spaced depths) at N=100 produced perfectly-structured recall:

> *"Here are all 5 passwords I was asked to remember, in the order they were
> encountered: 1. Password #1: OTTER41 (from topic_1); 2. Password #2: HORSE44
> (from topic_20); 3. Password #3: TIGER38 (from topic_40); 4. Password #4:
> MOOSE31 (from topic_60); 5. Password #5: MOOSE64 (from topic_80)."*

**Implication.** Single-fact and few-fact recall are well within modern
Claude's capability in the {5..200} range. H1's prediction of halftrace
> N=120 holds in the conservative direction (≫ N=200 with k=1; ≫ N=100
with k=5). To produce a halftrace at the intended N axis the task needs
qualitatively harder retention pressure: many more plants, distractors,
semantic interference between codewords, or recall under cognitive load.

### `instruction_decay`

Three rule designs tested, all on `find_and_synthesise` with `--serial`:

#### Rule v1: buried marker `[[END]]`

The marker requirement sat at the end of the system prompt, after the task
description. Sonnet 4.6 scored 0.0 on every cell — agent dropped the rule
from turn 1.

#### Rule v2: front-loaded marker `[[END]]`

Same marker rule, hoisted to the top of the system prompt with an explicit
format example.

| Model | N=5 | N=10 | N=25 |
| --- | --- | --- | --- |
| `claude-sonnet-4-6` | 0.833, 0.167, 0.167 | 0.091, 0.000, 0.818 | 0.962, 0.962, 0.962 |
| `claude-haiku-4-5` | 0.000, 0.000, 0.000 | 0.000, 0.000, 0.000 | 0.000, 0.000, 0.000 |

Sonnet 4.6 shows wild inter-rep variance — at N=10 one rep scored 0.091
and another 0.818 with the same prompt and seed structure. The first text
turn always violates the rule; after the first tool-result round trip,
the agent either commits to the rule for the rest of the trajectory or
abandons it entirely. Haiku 4.5 ignored the rule completely at every N.

#### Rule v3: incrementing counter `STATUS: <N>`

Hypothesis: a static literal is too cheap to drop or maintain; an
incrementing counter forces state-tracking and should produce gradual
drift. Tested at N={5, 7, 10, 15, 20, 25} with 2 reps each on Sonnet 4.6.

| N | rep=0 | rep=1 |
| --- | --- | --- |
| 5 | 1.000 | 1.000 |
| 7 | 0.250 | 1.000 |
| 10 | 0.182 | 1.000 |
| 15 | 1.000 | 1.000 |
| 20 | 0.095 | 1.000 |
| 25 | 1.000 | 1.000 |

Same bimodal pattern. At N=7/10/20 one rep complies perfectly and the
other drops the rule from turn 4 onward. At N=5/15/25 both reps comply.
There is no smooth N-dependence; the score is a per-trajectory coinflip.
A single earlier pilot at N=10 with 3 reps happened to draw three
"abandon" outcomes (all scoring 0.182), which produced a spurious
halftrace estimate of 8.06 with a degenerate CI — instructive about how
easily under-sampled bimodal data masquerades as a decay curve.

**Implication.** Across three rule designs and two models, instruction
adherence on `find_and_synthesise` is a coinflip-per-trajectory, not a
gradual decay. The probe is mechanically sound — it correctly captures
adherence — but the failure mode is binary: either the agent commits to
the rule after its first tool-result round trip or it never engages.
"Halftrace" as currently defined assumes smoothness that this failure
shape does not have.

### `tool_repetition`

Probe scores the fraction of tool calls that are unique by `(name, args)`.
Re-scored on every existing pilot trajectory (60 total across all pilot
files): **score 1.0 in every cell.** Sonnet 4.6 and Haiku 4.5 never
repeated a tool call in any trajectory.

**Implication.** `find_and_synthesise` gives the agent the complete topic
list upfront and instructs sequential lookup. There is no natural
opportunity for repetition. The probe is mechanically correct; the task
forecloses the failure mode.

### `narration_substitution`

Probe flags assistant text turns that mention a tool name without emitting
a corresponding `tool_call` in the same turn. Re-scored on every existing
pilot trajectory: **score 1.0 in every cell.** With `--serial`, every
text-bearing assistant turn also carries a tool call, so the substitution
signature never appears.

**Implication.** Same as `tool_repetition`: the task design forecloses
the failure mode. To elicit substitution we would need a task where the
agent might *choose* to skip work — long required outputs, no
upfront-completeness signal, ambiguous completion criteria.

## Methodological learnings

### Parallel tool use defeats trajectory-length scaling

Sonnet 4.6 emits all N `lookup` calls as a single parallel `tool_use`
response unless `tool_choice.disable_parallel_tool_use=true` is set.
Plant and recall annotations end up adjacent in the same user message,
so the "N tool calls" axis does not produce N rounds of context
accumulation. **The full study must run with `--serial` or an equivalent
control.**

### Prompt caching is essential at scale

The deep pilot at N=200 cost $4.03 for 9 trajectories. Of 7.4M total
input tokens, 7.2M were cache reads. Without prefix caching the same
pilot would have cost an order of magnitude more. Haiku's prefix at N≤25
is below its 4096-token cache threshold and does not benefit.

### Bimodal probes need many more reps than smooth probes

The pre-registered design called for 5 reps per cell, assuming smooth
decay. For bimodal probes (anything in the `instruction_decay` family)
that is wildly insufficient: 3 reps can land on all "abandon" outcomes
purely by chance and produce a halftrace estimate (8.06 in our case)
that vanishes with two more reps. Bimodal probes need 20+ reps per cell
to estimate the underlying commit-rate, not the halftrace.

### "Fail-then-comply" is a distinct failure mode

Every `instruction_decay` compliant trajectory shares the same shape:
turn 2 (the agent's first text turn, before any tool output) violates
the rule; every subsequent text turn complies. The agent reads the
system prompt instruction but defers rule-engagement until it has at
least one round of grounded interaction. This deserves its own probe.

### The same model + seed is not bit-deterministic across API calls

In the first counter-rule pilot, three reps at N=10 all scored 0.182.
In a second pilot the same prompt and same seed at N=10 once scored
0.182 (matching) and once scored 1.0 (different). Even at low effort
settings, the API does not guarantee bit-identical sampling. Treat
within-rep "determinism" as a draw, not a guarantee.

## Cost ledger

| Pilot | Spec | Cost |
| --- | --- | --- |
| Smoke | Sonnet 4.6, N=5, reps=1, parallel | $0.012 |
| First | Sonnet 4.6, N={5,10,25}, reps=3, parallel | $0.180 |
| Serial | Sonnet 4.6, N={5,10,25}, reps=3, serial | $0.440 |
| Deep | Sonnet 4.6, N={50,100,200}, reps=3, serial | $4.030 |
| Instr v1 | Sonnet 4.6, marker rule buried | $0.460 |
| Instr v2 | Sonnet 4.6, marker rule front-loaded | $0.450 |
| Haiku | Haiku 4.5, marker rule front-loaded | $0.250 |
| Multi-plant | Sonnet 4.6, k=5 plants, N={10,25,100} | $1.393 |
| Counter | Sonnet 4.6, counter rule, reps=3 | $0.436 |
| Counter-scan | Sonnet 4.6, counter rule, fine N, reps=2 | $0.610 |
| **Total** | | **$8.26** |

Well under the £200 cap from `HYPOTHESES.md` § Stopping conditions.
Per-trajectory reference cost at N=200 with caching: ~$0.45 on Sonnet 4.6.

## What's needed before the full study runs

The pilot phase has done its job: it surfaced concrete design issues
that make running the pre-registered 120-trajectory study a poor use of
budget without first addressing the following.

1. **Task redesign is the real bottleneck, not probe design.** All four
   implemented probes are mechanically correct. Three of them score 1.0
   universally because `find_and_synthesise` does not elicit the failure
   modes they measure. The fourth (`instruction_decay`) is bimodal
   because the task is short enough and well-specified enough that the
   agent's "comply or not" decision is taken once and stable. A task
   that elicits decay must be qualitatively different: ambiguous
   instructions, partial information, conflicting constraints, or
   open-ended completion criteria.
2. **The halftrace concept may need refinement for modern models.** The
   "smooth decay" assumption fits older or weaker models that gradually
   forget. Modern frontier Claude exhibits a commit-or-don't pattern on
   short, well-specified tasks. A useful diagnostic for this regime
   measures *commit probability* (fraction of trajectories that follow
   the rule end-to-end), not *decay rate*.
3. **Implement `premature_termination`** to complete the probe set, and
   re-score existing trajectories with it. If it shows 1.0 universally
   too, the task-redesign claim is unambiguously the only path forward.
4. **OpenAI adapter** for the GPT-4o / GPT-4.1 arms of the study. Mechanically
   a port of the Anthropic adapter; only worth building once tasks elicit
   decay on Anthropic models.
5. **Pre-register the post-pilot probe/task changes.** The original
   hypotheses assumed smooth decay; what we found is genuinely different
   and merits an updated pre-registration before scaling.

## What worked first try

For completeness — these components did not surface issues during the
pilot phase:

- Trajectory schema (pydantic, append-only, JSONL round-trip)
- Probe protocol and `Score` model
- `fit_halftrace` linear interpolation + bootstrap CI
- Anthropic adapter manual tool-use loop including `end_turn` termination,
  `stop_reason` recording, prompt caching, and `--serial` mode
- CLI argument parsing, dry-run mode, multi-probe scoring, and per-probe
  halftrace fitting
- `n_plants` parameter on `find_and_synthesise`
- Plural `fact_ids` schema for multi-plant recall annotations
