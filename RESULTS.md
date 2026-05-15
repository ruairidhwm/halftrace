# Pilot results

Pre-pilot status, last updated 2026-05-14. The pilots described here are
scoped to validate the halftrace measurement pipeline against real model
behaviour at small N, before committing to the full 120-trajectory study
laid out in `HYPOTHESES.md`. They are not the full study.

## Summary

Twelve pilot runs against the Anthropic API (~64 trajectories, $8.77 of
spend) end-to-end validated the pipeline: `Task` → adapter → `Trajectory`
→ probe → `fit_halftrace` → CLI. All four `find_and_synthesise` probes
are implemented (`state_amnesia`, `instruction_decay`, `tool_repetition`,
`narration_substitution`) and have been measured against `claude-sonnet-4-6`
and `claude-haiku-4-5` across N values from 5 to 200, including multi-plant
state_amnesia variants, three different `instruction_decay` rule designs,
and an adversarial discovery-mode task that forces the agent to maintain
its own `seen` list.

**No probe produced a halftrace in any configuration.** This is itself the
finding worth recording. The consistent pattern across the design space:

- `state_amnesia`, `tool_repetition`, `narration_substitution`: flat 1.0
  everywhere — including under multi-plant retention, doubled trajectory
  length, and adversarial state-tracking pressure. The failure modes are
  not exercised by *any* version of `find_and_synthesise` we tried.
- `instruction_decay`: **categorical, not gradient.** Three patterns
  observed depending on rule and task variant, but all categorical:
  - *Bimodal commit-or-abandon* (sequential task, marker or counter rules):
    coinflip per trajectory, no N dependence.
  - *Categorical alternation* (discovery-mode task, counter rule): the
    agent applies the rule to one turn-type and drops it on another,
    yielding a 1/2 compliance ratio that is independent of N.

**Modern Claude (Sonnet 4.6) treats format rules as *categorical
decisions per turn-type*, not as gradients over trajectory length.**
Compliance is determined by the agent's classification of "which kind of
turn this is", not by accumulated context pressure. Across twelve pilots,
~64 trajectories, three rule designs, two task modes, two models, and N
from 5 to 200, we have observed zero instances of gradual decay on any
probe. The halftrace concept as defined in `HYPOTHESES.md` assumes smooth
degradation, which is the wrong shape for the failure modes we observe.

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

#### Rule v3: incrementing counter `STATUS: <N>` (sequential task)

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

#### Rule v3 + discovery-mode task: categorical alternation

Same counter rule, but using the discovery-mode variant of the task: the
agent must call `discover_next(seen=[...])` to reveal each topic before
looking it up, doubling the trajectory length and adding bookkeeping
pressure. Tested at N={10, 25} with 2 reps each on Sonnet 4.6.

| N | rep=0 | rep=1 | trajectory length |
| --- | --- | --- | --- |
| 10 | 0.545 | 0.545 | 22 assistant turns |
| 25 | 0.519 | 1.000 | 52 assistant turns |

The intermediate scores (0.545, 0.519) are not partial decay — they
encode a **stable alternating pattern**: bit-string `1010101010...11`
across all observed turns, where the agent applies `STATUS:` to
`discover_next` narration turns and drops it on the immediately-
following `lookup` narration turns. The compliance fraction converges
to 1/2 asymptotically (12/22 at N=10, 27/52 at N=25) because exactly
half the text turns are each type. Same model, same rule, same prompt:
the categorisation is stable.

Example: at the start of an N=25 trajectory the agent emits
`"STATUS: 1\nI'll start by discovering the first topic..."` (compliant)
and one turn later `"Noted! Password #1: MOOSE65 — I'll remember this.\nNow I'll look up topic_25."`
(non-compliant). The first is a discovery announcement; the second is a
lookup acknowledgement. The agent treats them as different rule classes.

**Implication.** Across three rule designs, two task variants, and two
models, instruction adherence on `find_and_synthesise` is *categorical
per turn-type*, never gradual. The agent classifies turn function and
applies rules conditionally — but consistently within each class. The
probe is mechanically sound; the "halftrace" notion that compliance
gradually erodes with context does not match the observed behaviour.

### `tool_repetition`

Probe scores the fraction of tool calls that are unique by `(name, args)`.
Re-scored on every pilot trajectory (~64 total): **score 1.0 in every cell.**
Sonnet 4.6 and Haiku 4.5 never repeated a tool call in any trajectory,
including the discovery-mode pilots where the agent had to maintain its
own `seen` list across 22-52 turns.

**Implication.** `find_and_synthesise` (sequential *or* discovery) does
not give Sonnet 4.6 enough cognitive pressure to repeat. The probe is
mechanically correct; even adversarial state-tracking is within the
model's capability at this N range.

### `narration_substitution`

Probe flags assistant text turns that mention a tool name without emitting
a corresponding `tool_call` in the same turn. Re-scored on every pilot
trajectory: **score 1.0 in every cell.** With `--serial`, every
text-bearing assistant turn also carries a tool call, so the substitution
signature never appears. Discovery mode does not change this.

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
| Discovery | Sonnet 4.6, discovery-mode, counter rule, N={10,25}, reps=2 | $0.507 |
| **Total** | | **$8.77** |

Well under the £200 cap from `HYPOTHESES.md` § Stopping conditions.
Per-trajectory reference cost at N=200 with caching: ~$0.45 on Sonnet 4.6.

## What's needed before the full study runs

The pilot phase has done its job: it has shown that the failure modes
the project was originally designed to measure do not manifest as
gradient decay on modern Claude in any agentic task structure we tried.
The pre-registered 120-trajectory study would, on current evidence,
produce flat 1.0 cells and zero halftraces — running it is not worth
the budget without first addressing the framing issue.

1. **The halftrace concept itself needs reframing.** "Smooth decay"
   fits older or weaker models that gradually forget under context
   pressure. Modern frontier Claude exhibits *categorical* compliance
   patterns: commit-or-abandon at the trajectory level, or stable
   per-turn-type rule application within a trajectory. A useful
   diagnostic for this regime measures the *shape* of the compliance
   pattern (perfect / bimodal / categorical / decaying), not a single
   halftrace number that would be undefined for three of those four
   shapes. See "Tool redesign" below.
2. **Task design has a hard ceiling on this model class.** We tried
   sequential, multi-plant, and discovery-mode variants of
   `find_and_synthesise`. Each added cognitive pressure; none produced
   gradual decay on `state_amnesia`, `tool_repetition`, or
   `narration_substitution`. To elicit those failure modes on modern
   Claude likely requires a qualitatively different task (open-ended
   completion criteria, ambiguous information, conflicting constraints)
   — a substantial design investment that should follow rather than
   precede the framework reframe.
3. **Implement `premature_termination`** to complete the probe set. On
   current evidence it will score 1.0 universally on existing data,
   reinforcing the task-foreclosure claim.
4. **Cross-model comparison via the same framework.** Once the tool is
   reframed around *compliance shape* rather than *halftrace*, Haiku
   4.5 / Opus 4.7 / GPT comparisons become meaningful — they measure
   which shapes each model produces on which rules.

## Tool redesign

The empirical findings argue for a specific reframe of what halftrace
*is* and what it *outputs*. The components that survive intact:

- `Trajectory`, `Probe`, `Score` — per-trajectory scoring is sound.
- The four probes themselves — they correctly measure compliance per
  trajectory.
- Adapter, task scaffolding, CLI runner — the data-collection pipeline
  works.

The component that does not survive: `fit_halftrace` and its halftrace
output. Linear interpolation over per-N means assumes a monotone curve,
which the data does not show. In its place, a useful diagnostic emits
a *compliance profile* per (agent, task, probe) cell:

- **Shape classification.** Detect whether the per-N scores look like
  *perfect*, *abandoned*, *bimodal*, *categorical-partial*, *gradient*
  (the case the original framework assumed), or *unclassified*.
- **Per-N mean and within-N variance.** Variance above ~0.3 within a
  cell is the bimodal signature; below ~0.05 with intermediate means
  is the categorical signature.
- **Per-turn-type breakdown.** For `instruction_decay`-like probes,
  partition compliance by which kind of turn produced it (lookup vs
  discovery, action vs acknowledgment, etc.). The categorical
  alternation finding emerges directly from this.
- **Commit probability.** Fraction of trajectories that follow the
  rule end-to-end. Replaces "halftrace" as the headline scalar for
  bimodal probes.

This reframe keeps the entire data-collection pipeline intact, replaces
the curve-fitting layer with a classification + per-segment-summary
layer, and makes the diagnostic actually useful for modern models on
agentic tasks: it tells the agent developer *which shape* their model's
compliance has on a given rule, not a single number that hides the
underlying pattern.

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
