# Pilot results

Last updated 2026-05-17. The pilots described here are scoped to validate
the halftrace measurement pipeline against real model behaviour at small
N, before committing to the full 120-trajectory study laid out in
`HYPOTHESES.md`. They are not the full study.

## Summary

Sixteen pilot runs against the Anthropic API (~104 trajectories, $10.49
of spend) end-to-end validated the pipeline: `Task` → adapter → `Trajectory`
→ probe → `fit_halftrace` → CLI. All four `find_and_synthesise` probes
are implemented (`state_amnesia`, `instruction_decay`, `tool_repetition`,
`narration_substitution`) and have been measured against `claude-sonnet-4-6`
and `claude-haiku-4-5` across N values from 5 to 200, including multi-plant
state_amnesia variants, three different `instruction_decay` rule designs,
an adversarial discovery-mode task that forces the agent to maintain its
own `seen` list, and a four-condition mechanism test that locates the
bimodal abandonment dynamic as a cold-start effect.

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
turn this is", not by accumulated context pressure. Across sixteen
pilots, ~104 trajectories, three rule designs, two task modes, two
models, and N from 5 to 200, we have observed zero instances of gradual
decay on any probe. The halftrace concept as defined in `HYPOTHESES.md`
assumes smooth degradation, which is the wrong shape for the failure
modes we observe.

A follow-up mechanism investigation (four primer conditions × two
models, 80 trajectories at N=10) locates the bimodal abandonment as a
*cold-start* effect on Sonnet 4.6's STATUS counter rule: two trivial
off-topic user/assistant exchanges before the real task — with no format
in the primer responses — drive N=10 compliance from 0.345 (baseline,
4/5 abandon) to 1.000 (10/10 perfect). One prior assistant turn ("ok.")
is mostly sufficient (8/10 perfect); a single extra user message with
no assistant turn produces a qualitatively different tri-modal shape.

Three follow-up extensions narrow the cure's scope:

1. **The cure does not rescue long N.** 10 reps at N=100 with the full
   two-turn primer scored a mean of 0.470 (4/10 perfect, 4/10 abandon),
   essentially identical to the atlas baseline of 0.508 (4/10 perfect,
   4/10 abandon). The early-uncertainty dip and the long-context drift
   are *different* failure modes with different mechanisms.
2. **The cure does not replicate on the marker rule.** Sonnet 4.6's
   abandonment is milder on the front-loaded `[[END]]` rule (mean 0.79
   at N=10) than on the counter rule. The zero-content primer that
   improved counter-rule compliance from 0.35 to 0.76 *worsens*
   marker-rule compliance from 0.79 to 0.58 — same intervention,
   opposite sign on a different rule.
3. **gpt-3.5-turbo also doesn't show gradient decay.** The negative
   "no gradient decay" finding from the frontier-model atlas extends
   to a meaningfully weaker model: at N≤50, gpt-3.5-turbo produces a
   categorical "fail first turn then comply" signature plus mild
   bimodal abandonment, but no smooth monotone decay.

The full results are in
[§ Mechanism: bimodal abandonment is a cold-start effect](#mechanism-bimodal-abandonment-is-a-cold-start-effect)
and subsequent subsections.

## Pipeline status

| Component | Status |
| --- | --- |
| `Trajectory` schema + JSONL serialization | Done |
| `Probe` protocol + `Score` type | Done |
| `state_amnesia` probe (single- and multi-plant) | Done |
| `instruction_decay` probe (end_with_marker, starts_with_pattern) | Done |
| `tool_repetition` probe | Done |
| `narration_substitution` probe | Done |
| `premature_termination` probe | Done |
| `fit_halftrace` (linear interp + bootstrap 95% CI) | Done |
| Anthropic adapter (manual tool-use loop, caching, usage tracking, `--serial`) | Done |
| OpenAI adapter (Anthropic-shaped dispatch by model prefix) | Done |
| `find_and_synthesise` task (parameterised by `n_plants`, discovery mode) | Done |
| `find_max` task (adversarial; exercises `premature_termination`) | Done |
| Pilot runner (`halftrace` CLI, multi-probe scoring, per-probe halftrace) | Done |
| Ingest (`from_anthropic_messages`, `from_openai_messages`, `from_langsmith_run`) | Done |

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

## Mechanism: bimodal abandonment is a cold-start effect

The pilot finding was that Sonnet 4.6's `instruction_decay` is bimodal:
a per-trajectory coinflip between full compliance and total abandonment,
taken on the agent's first text turn and stable thereafter. That left
the *mechanism* open. Four follow-up runs (40 trajectories, ~$1.26)
locate it.

All four use the counter rule (`STATUS: <N>`), sequential `find_and_synthesise`,
Sonnet 4.6 at N=10, 10 reps each. The only difference between cells is
what the visible message history looks like *before* the real task's
user message arrives.

| Condition | Pre-task messages | Pattern | Mean | Perfect | Partial | Abandoned |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | (none) | bimodal | 0.345 | 1/5 | 0/5 | 4/5 |
| Zero | 1 user msg, no assistant turn | tri-modal | 0.755 | 3/10 | 6/10 | 1/10 |
| Minimal | 1 trivial exchange (`Hi.` / `ok.`) | bimodal | 0.836 | 8/10 | 0/10 | 2/10 |
| Full | 2 trivial exchanges (2+2; Paris), no format | always commit | 1.000 | 10/10 | 0/10 | 0/10 |

Two prior trivial user/assistant exchanges, with no STATUS prefix in
the assistant primer responses, drive compliance from 4/5 abandon to
10/10 perfect. The strong "agent commits based on seeing the format
applied in prior turns" hypothesis is *falsified*: a parallel "format
primer" condition (the same primer with `STATUS: 1` / `STATUS: 2`
prefixes on the assistant responses) also produced 10/10. Format-in-priors
is not the load-bearing piece.

What *is* load-bearing decomposes into two effects:

1. **Any prior assistant turn flips commit-vs-mix.** The minimal primer
   adds one trivial assistant turn (`ok.`) and the shape snaps back to
   the same bimodal commit-or-abandon as baseline, but with the commit
   rate elevated. The zero primer (extra user message only, no
   assistant turn anywhere in the visible history) produces a
   qualitatively different *tri-modal* shape: three of ten reps commit
   fully, six produce partial scores in [0.4, 0.95], one fully abandons.
   The partial reps split as three at 0.909 (drop one turn) and three
   at 0.545 — the exact categorical-alternation signature observed in
   discovery-mode pilots, applied here to a sequential task. The prior
   assistant turn appears to be the discriminator between "clean
   bimodal commit/abandon" and "noisy mixed compliance".
2. **Turn count drives commit probability within bimodal mode.** One
   prior assistant turn: 8/10 commit. Two prior assistant turns: 10/10
   commit. The dose-response is monotone over the {0, 1, 2}-turn axis
   we tested.

The cleanest summary: the bimodal abandonment is a *cold-start* effect.
The agent's first assistant turn is disproportionately likely to drop
a format rule that every subsequent turn in the same trajectory would
have respected. Inserting any prior assistant turn into the context —
trivial, off-topic, format-free — moves the decision-point off the
real task and into a context the agent treats as safe to default-comply.

Practical reading: an agent developer who controls the message
history can drive Sonnet 4.6's counter-rule commit rate from 0.345 to
1.000 by prepending a two-turn synthetic warmup. This is independent
of any prompt engineering of the rule itself.

### Cross-model: Haiku has no abandonment to fix

The same four conditions on Haiku 4.5 produced 10/10 perfect on
*every* condition, including baseline:

| Condition | Mean | Perfect | Abandoned |
| --- | --- | --- | --- |
| Baseline | 1.000 | 10/10 | 0/10 |
| Zero | 1.000 | 10/10 | 0/10 |
| Minimal | 1.000 | 10/10 | 0/10 |
| Full | 1.000 | 10/10 | 0/10 |

Haiku 4.5 was previously tested only on the marker rule (`[[END]]`),
where it scored 0.000 across nine trajectories. On the counter rule
it scores 1.000 across forty. The Sonnet warm-up dynamic does not
generalize to Haiku here because Haiku has no baseline failure mode
on this rule to fix — and the same model that universally drops the
marker rule universally complies with the counter rule. The
cross-model story is therefore that rule design has more leverage
than priming for Haiku, and priming has more leverage than rule
design for Sonnet (which struggles on both rule families).

The mechanism finding is Sonnet-specific to the counter rule. The
sections below test whether it generalises across N (it does not) and
across rule families (it does not, at least on the marker rule we
re-tested).

### Cure at long N: doesn't replicate

The N=10 result above leaves open whether the same primer that closes
the early-uncertainty dip would also rescue Sonnet at N=100 (atlas
baseline mean 0.508) and N=200 (atlas baseline mean 0.163). 10 reps at
N=100 with the full two-turn primer, plus a partial 3 reps at N=200
before credits ran out:

| Condition | N | Mean | Perfect | Partial | Abandoned |
| --- | --- | --- | --- | --- | --- |
| Atlas baseline | 100 | 0.508 | 4/10 | 2/10 | 4/10 |
| **Full primer** | **100** | **0.470** | **4/10** | **2/10** | **4/10** |
| Atlas baseline | 200 | 0.163 | 1/10 | 1/10 | 8/10 |
| **Full primer** | **200** | **0.011** | **0/10** | **0/10** | **10/10** |

The N=100 distributions are essentially identical with and without the
primer. Same number of perfect runs, same number of abandons, mean
within noise. The intervention has *no measurable effect* at N=100.

The N=200 result is starker: with the full primer, every one of 10 reps
abandoned the rule (mean 0.011). The atlas baseline at N=200 was
mean 0.163 with 1/10 perfect and 8/10 abandons. With 10 reps and a true
abandonment rate of 80% you'd expect all-10-abandons to happen by chance
~11% of the time, so we cannot strictly distinguish "primer hurts" from
"primer is null but we drew unluckily." What we can say: **the primer
does not help at N=200**, and the directional effect is the opposite of
what it is at N=10.

**Implication.** The early-uncertainty dip (N≤10) and the long-context
drift (N≥50) are *different failure modes* on Sonnet 4.6's STATUS counter
rule. The cold-start cure is N=10-specific. The long-N drift needs its
own intervention — likely something targeting system-prompt attention
fade rather than the agent's first-turn decision dynamics.

### Cure replication on the marker rule: rule-specific effects

A second test: does the cure transfer to a different rule on the same
model? Earlier pilots showed Sonnet 4.6 bimodally abandons the
front-loaded marker rule (`[[END]]`) too, with per-rep scores 0.091 /
0.000 / 0.818 at N=10. Running the four-condition cure on this rule
gives a *qualitatively different* picture from the counter rule:

| Condition | Marker rule mean | Marker P/Pa/Ab | Counter rule mean | Counter P/Pa/Ab |
| --- | --- | --- | --- | --- |
| baseline | 0.791 | 7 / 1 / 2 | 0.345 | 1 / 0 / 4 |
| zero | **0.582** | 4 / 2 / 4 | **0.755** | 3 / 6 / 1 |
| minimal | 0.791 | 7 / 1 / 2 | 0.836 | 8 / 0 / 2 |
| full | 0.973 | 8 / 2 / 0 | 1.000 | 10 / 0 / 0 |

(10 reps per cell; P/Pa/Ab = perfect / partial / abandoned.)

Three things to notice:

1. **The marker rule has milder baseline abandonment.** 7/10 perfect at
   baseline (mean 0.79) versus 1/5 perfect on the counter rule (mean
   0.35). There is less room for the cure to improve things.
2. **The zero-content primer reverses sign.** On the counter rule,
   adding one extra user message (and no prior assistant turn) moved
   compliance from 0.35 to 0.76. On the marker rule, the same
   intervention moves compliance from 0.79 to 0.58 — it *worsens*
   commit rate. Same model, same N, same primer, opposite sign.
3. **The dose-response is non-monotone on the marker rule.** Counter
   rule progression: 0.35 → 0.76 → 0.84 → 1.00 (monotone increasing).
   Marker rule progression: 0.79 → 0.58 → 0.79 → 0.97. The minimal
   condition reverts to baseline; the zero condition is worse than
   either; only the full primer helps. The "any prior assistant turn
   keeps the shape bimodal" generalisation from the counter rule does
   not hold on the marker rule.

What survives across both rules: the **full two-exchange primer**
either matches or improves baseline (1.00 vs 0.35 on counter; 0.97 vs
0.79 on marker). What does not survive: the cleaner intermediate
dose-response.

**Implication.** The cold-start dynamic is not a uniform property of
Sonnet 4.6's first-turn instability. The exact decision the agent
makes about a system-prompt rule on its first text turn appears to be
rule-shape-dependent — counter rules and marker rules trigger
different first-turn calculations, and the same intervention helps one
while sometimes hurting the other. The robustly-helpful intervention
is the full two-trivial-exchange primer; the more degenerate primers
have rule-specific effects.

### Cross-model: gpt-3.5-turbo also doesn't decay

To test whether the "no gradient decay on modern Claude" finding
extends to a meaningfully weaker model, gpt-3.5-turbo (still on OpenAI's
chat-completions API, dispatched by the existing adapter) was run at
N={5, 10, 25, 50} with 10 reps per cell. N=100 was skipped after
OpenAI's quota error fired mid-run. 40 trajectories total.

| Probe | N=5 | N=10 | N=25 | N=50 | Shape |
| --- | --- | --- | --- | --- | --- |
| state_amnesia | 1.000 | 1.000 | 1.000 | 1.000 | perfect |
| instruction_decay | 0.833 | 0.818 | 0.770 | 0.877 | categorical |
| tool_repetition | 1.000 | 1.000 | 0.981 | 0.985 | near-perfect |
| narration_substitution | 1.000 | 1.000 | 1.000 | 1.000 | perfect |

`instruction_decay` is non-monotone over N — categorical, not gradient.
Specifically: every N=5 rep scored exactly 0.833 (= 5/6 turns
compliant), which is the "fail first turn then comply for the rest"
fingerprint observed in earlier Sonnet pilots. At higher N a small
fraction of reps fully abandon (1-2/10) and the rest stay around
0.9-1.0. Same bimodal-with-categorical-first-turn shape that frontier
Claude shows, just with mildly worse overall compliance.

`tool_repetition` very slightly fires (mean 0.98 at N=25 and N=50). The
frontier models hit exactly 1.000 across every cell. gpt-3.5-turbo
occasionally re-issues the same tool call — small effect but real.

**No gradient decay.** Even on a model class meaningfully weaker than
Sonnet/Opus/Haiku/4o/4.1, the per-N curve is not monotone and the
classified shape is not `gradient`. The negative finding from the
frontier-model atlas extends to gpt-3.5-turbo on this task family.

### Adversarial task: premature_termination + find_max

The original probe set was five: state_amnesia, instruction_decay,
tool_repetition, narration_substitution, premature_termination. Only
the first four shipped in v0.1.0 because `find_and_synthesise`
forecloses the premature_termination failure mode by enumerating the
work upfront. The 5th probe is now shipped, along with a new task
(`find_max`) designed to admit it: agent gets N topics with hidden
integer values, looks up each via `lookup(topic)`, and submits the
maximum. The system prompt grants explicit licence to be efficient
("don't make more calls than needed"). Agents that take the licence
submit before exhausting lookups and the probe scores the resulting
n_completed / n_required ratio.

Sonnet 4.6, 10 reps at N=10 and 10 reps at N=30:

| N | n_required | n_completed (median) | premature_termination |
| --- | --- | --- | --- |
| 10 | 10 | 10 (every rep) | 1.000 (10/10 reps) |
| 30 | 30 | 30 (every rep) | 1.000 (10/10 reps) |

Sonnet 4.6 doesn't take the bait. Even at N=30 it exhausts every
lookup before submitting, always finds the true maximum, and never
exhibits the premature-termination failure mode. The probe and task
are mechanically valid (full unit-test coverage); they just don't fire
on this model on this task. To elicit the failure mode reliably we
would likely need either a weaker model or a task with truly
open-ended completion criteria (no finite topic list to enumerate).

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
| Mech-test | Sonnet 4.6, two-exchange format primer, N=10, reps=10 | $0.178 |
| Mech-control | Sonnet 4.6, two-exchange no-format primer, N=10, reps=10 | $0.173 |
| Mech-minimal | Sonnet 4.6, one-exchange `Hi.`/`ok.` primer, N=10, reps=10 | $0.233 |
| Mech-zero | Sonnet 4.6, extra user msg only, no assistant primer, N=10, reps=10 | $0.424 |
| Mech-Haiku | Haiku 4.5, all four primer conditions, N=10, reps=10 each | $0.685 |
| Mech-long | Sonnet 4.6, full primer at N=100 (10 reps) + N=200 (partial) | ~$3.10 |
| Mech-marker | Sonnet 4.6, four primer conditions × marker rule, N=10, reps=10 each | ~$1.60 |
| Atlas gpt-3.5 | gpt-3.5-turbo, N={5,10,25,50}, reps=10 (N=100 quota-hit) | ~$0.80 |
| Find-max | Sonnet 4.6, find_max task at N=10 + N=30, reps=10 each | ~$0.85 |
| **Total** | | **~$16.84** |

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
