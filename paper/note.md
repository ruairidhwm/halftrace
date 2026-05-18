# Compliance is categorical, not gradient: a negative result from sixteen halftrace pilots on Claude Sonnet 4.6

**Ruairidh Wynne-McCorry**

*Last updated: 2026-05-17. Companion repository: [github.com/ruairidhwm/halftrace](https://github.com/ruairidhwm/halftrace).*

---

## Abstract

The "context rot" intuition predicts that LLM-based agents gradually degrade on long-horizon tasks, with different capabilities decaying at different rates. We pre-registered a study ([HYPOTHESES.md](../HYPOTHESES.md)) around fitting a *halftrace* — the trajectory length at which a capability hits 50% of baseline — for four failure modes (state amnesia, instruction decay, tool repetition, narration substitution) on Claude Sonnet 4.6 and Haiku 4.5. The atlas (~230 trajectories across five frontier models, N from 5 to 200, $40 in API spend) produced **zero gradient decay on any probe**. Compliance is instead *bimodal per trajectory* (commit-or-abandon, approximately a coinflip) or *categorical per turn-type* (stable per-turn-class application, independent of trajectory length). A four-condition mechanism investigation at N=10 (80 trajectories on Sonnet plus 40 on Haiku) initially located the bimodal abandonment as a *cold-start* effect: two trivial off-topic user/assistant exchanges before the real task — with no format in the primer responses — drive Sonnet's N=10 compliance from 0.345 to 1.000. Two follow-up extensions sharpen and narrow that mechanism. **The same primer has no measurable effect at N=100** (mean 0.470 with primer vs 0.508 baseline) — the early-uncertainty dip and the long-N drift are *different* failure modes. **The same primer produces a non-monotone dose-response on a different rule** (marker rule `[[END]]`), with the zero-content primer actually worsening compliance. The intervention is N=10-specific on the counter rule, partly rule-specific overall, and the full two-exchange primer is the only robustly helpful form. A cross-model sanity check on gpt-3.5-turbo (40 trajectories at N≤50) finds no gradient decay there either — the negative finding extends beyond frontier Claude. We release [halftrace](https://github.com/ruairidhwm/halftrace) reframed around shape classification, with the five-probe set complete (premature_termination + find_max ship in v0.1.1) and a new LangSmith trace ingest.

## 1. Background

Anecdotal reports of "context rot" — agents that work well at low trajectory length but fail in weird ways at higher N — motivated the original halftrace framework. The pre-registered hypothesis [H1] ordered four failure modes by predicted halftrace:

> 1. `instruction_decay` — halftrace under N=40 on most models
> 2. `narration_substitution` — halftrace between N=40 and N=80
> 3. `tool_repetition` — halftrace between N=60 and N=120
> 4. `state_amnesia` — halftrace above N=120 on most models

The expectation: each failure mode would produce a smooth monotonic decay curve over N, with a clean crossing of the 0.5 compliance threshold at the predicted point. The diagnostic instrument would fit that curve, emit the halftrace point, and provide a 95% bootstrap CI.

The pilot phase was scoped to validate this measurement pipeline at small N before committing to the full 120-trajectory pre-registered study. It did not validate the measurement: it falsified the underlying assumption that compliance gradients exist at all on this task family.

## 2. Method

### 2.1 Task

A synthetic agentic task, `find_and_synthesise`, parameterised by N (the number of canonical tool calls required). The agent is given N topics and instructed to look up each one via a `lookup` tool, then call `submit_summary` to end the task. Codewords are planted in lookup responses at evenly-spaced positions; a final lookup asks the agent to recall them. An adversarial discovery-mode variant requires the agent to maintain its own `seen` list via a `discover_next` tool, with the server validating that the seen list matches the revealed history.

### 2.2 Probes

Four probes ship with halftrace and score per trajectory:

- **state_amnesia.** Fraction of planted facts the agent correctly recalls. Annotations on plant/recall turns; the probe walks the trajectory and pairs each recall with the next assistant text turn.
- **instruction_decay.** Fraction of assistant text turns that satisfy a system-prompt rule. Three rule designs were tested: a literal end-of-response marker (`[[END]]`), a hoisted variant of the same marker, and an incrementing counter (`STATUS: <N>`).
- **tool_repetition.** Fraction of tool calls that are unique by `(name, args)`.
- **narration_substitution.** Fraction of assistant text turns that mention a tool name without emitting a corresponding tool call in the same turn.

Each probe returns a `Score` with `value ∈ [0, 1] ∪ {None}`. `value = None` means the probe could not score (no annotations present, no observations to make).

### 2.3 Models, N values, and rep counts

- **Models:** `claude-sonnet-4-6` (primary), `claude-haiku-4-5` (cross-model runs on marker rule and full four-condition mechanism replication on counter rule).
- **N values:** {5, 7, 10, 15, 20, 25, 50, 100, 200} — not all cells tested on all probe/model combinations.
- **Reps per cell:** 2-3 for the original probe pilots; **10** for each mechanism cell (§3.5).
- **Mandatory:** `tool_choice.disable_parallel_tool_use=true`. Without it, Sonnet 4.6 batches all N lookups into a single parallel `tool_use` response, collapsing the N axis entirely. The `--serial` flag was used for every pilot after this was discovered.

### 2.4 Mechanism conditions

Four primer conditions, each 10 reps at N=10 on counter rule, run on both Sonnet 4.6 and Haiku 4.5. The conditions differ only in what message history is visible before the real task's user message arrives:

- **Baseline:** no primer; the real-task user message is the first message after the system prompt.
- **Zero:** one extra user message (`"Hi."`) before the real-task user message; no prior assistant turn anywhere in the visible context.
- **Minimal:** one trivial exchange (`"Hi."` / `"ok."`) before the real-task user message; the assistant's primer reply contains no STATUS prefix.
- **Full:** two trivial off-topic exchanges (2+2, capital of France) before the real-task user message; assistant primer replies contain no STATUS prefix.

A separately-tested companion to "Full" (not included in the cross-model run) uses the same two-exchange structure but with assistant primer replies that *do* carry a `STATUS: 1` / `STATUS: 2` prefix. We compare the two to test whether prior-format-presence is doing the work or prior-assistant-turn-count is.

The same four conditions were also run on Sonnet 4.6 with the **marker rule** (front-loaded `[[END]]`) instead of the counter rule (§3.7), and at **N=100 and N=200** with the "full" primer to test whether the cure transfers to long context (§3.6).

### 2.5 Total spend

~$45 across the atlas, the original pilot phase, and the follow-up mechanism investigations combined. The full mechanism programme (four primer conditions × counter rule × Sonnet + four × Haiku + four × marker rule × Sonnet + long-N cure at N=100/200 + gpt-3.5-turbo atlas + find_max pilot) accounts for ~$8 of that total across ~280 follow-up trajectories. Per-trajectory cost reference at N=200 with prefix caching enabled: ~$0.45 on Sonnet 4.6, ~$2.25 on Opus 4.7. Full ledger in [RESULTS.md §Cost ledger](../RESULTS.md#cost-ledger).

## 3. Results

### 3.1 `state_amnesia`: flat 1.0 across the design space

| Variant | Model | N=5 | N=10 | N=25 | N=50 | N=100 | N=200 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| k=1 plant | Sonnet 4.6 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| k=1 plant | Haiku 4.5 | 1.000 | 1.000 | 1.000 | — | — | — |
| k=5 plants | Sonnet 4.6 | — | 1.000 | 1.000 | — | 1.000 | — |

Every cell scored 1.0. The single-plant probe correctly captured a planted codeword across 400+ intervening trajectory turns at N=200 under forced serialization. The multi-plant probe (k=5 codewords at evenly-spaced depths) at N=100 produced perfectly-structured recall:

> *"Here are all 5 passwords I was asked to remember, in the order they were encountered: 1. Password #1: OTTER41 (from topic_1); 2. Password #2: HORSE44 (from topic_20); 3. Password #3: TIGER38 (from topic_40); 4. Password #4: MOOSE31 (from topic_60); 5. Password #5: MOOSE64 (from topic_80)."*

H1's prediction that `state_amnesia` halftrace > N=120 holds in the conservative direction (≫ N=200 with k=1, ≫ N=100 with k=5). The probe in its current form does not separate modern Claude models on single-fact recall in this trajectory-length range.

### 3.2 `instruction_decay`: bimodal commit-or-abandon

Three rule designs all produced the same pattern on Sonnet 4.6: a coinflip per trajectory between full compliance and total abandonment, with the choice taken on turn 4 (the agent's first text turn after a tool result) and stable for the rest of the trajectory.

Front-loaded marker rule, per-rep scores:

| Model | N=5 | N=10 | N=25 |
| --- | --- | --- | --- |
| Sonnet 4.6 | 0.833, 0.167, 0.167 | 0.091, 0.000, 0.818 | 0.962, 0.962, 0.962 |
| Haiku 4.5 | 0.000, 0.000, 0.000 | 0.000, 0.000, 0.000 | 0.000, 0.000, 0.000 |

Counter rule on Sonnet 4.6, fine-N scan (2 reps each):

| N | rep=0 | rep=1 |
| --- | --- | --- |
| 5 | 1.000 | 1.000 |
| 7 | 0.250 | 1.000 |
| 10 | 0.182 | 1.000 |
| 15 | 1.000 | 1.000 |
| 20 | 0.095 | 1.000 |
| 25 | 1.000 | 1.000 |

The bimodality is not noise: an earlier pilot at N=10 with 3 reps happened to draw three "abandon" outcomes and produced a halftrace estimate of 8.06 with a degenerate confidence interval. A fresh run at the same N drew a 0.182 and a 1.000, illustrating both the underlying stochasticity (API draws are not bit-deterministic across calls even at low effort) and the danger of under-sampled bimodal data masquerading as a decay curve. Haiku 4.5 across every rep at every N ignored the rule entirely, hitting the `abandoned` shape rather than the `bimodal` one.

### 3.3 `instruction_decay`: categorical alternation under task pressure

The discovery-mode task variant (agent maintains its own `seen` list, doubling trajectory length and adding bookkeeping pressure) produced a different but equally non-gradient pattern: stable alternating compliance, independent of N.

| N | rep=0 | rep=1 | trajectory length |
| --- | --- | --- | --- |
| 10 | 0.545 | 0.545 | 22 assistant turns |
| 25 | 0.519 | 1.000 | 52 assistant turns |

The intermediate scores encode the bit-string `1010101010...11` across all assistant text turns: the agent applies the `STATUS:` prefix to `discover_next` narration turns and drops it on the immediately-following `lookup` narration turns, even though the rule is stated universally. Compliance fraction converges to 1/2 asymptotically because exactly half the text turns are each type.

Example turns from the same trajectory:

- *Compliant (turn 2):* `"STATUS: 1\nI'll start by discovering the first topic..."`
- *Non-compliant (turn 4):* `"Noted! Password #1: MOOSE65 — I'll remember this.\nNow I'll look up topic_25."`

This is not partial decay. It is a stable rule-class discrimination: the agent has decided that the format requirement applies to "investigation" turns but not to "retrieval acknowledgement" turns. The discrimination is consistent within and across trajectories.

### 3.4 `tool_repetition` and `narration_substitution`: foreclosed by task design

Both probes scored 1.0 in every cell across every pilot, on both models, including under multi-plant load and discovery-mode bookkeeping pressure. Sonnet 4.6 and Haiku 4.5 never repeated a tool call, never narrated a tool name in text without emitting a corresponding tool_call, even at N=200 in serial mode.

The probes are mechanically correct. The task design forecloses both failure modes: `find_and_synthesise` gives the agent a deterministic enumeration of work and the discovery variant adds pressure but never makes repetition or narration shortcuts strategically preferable. Eliciting these failure modes likely requires a qualitatively different task — open-ended completion criteria, ambiguous information, conflicting constraints.

### 3.5 Mechanism: bimodal abandonment at N=10 is a cold-start effect

The pilot finding that Sonnet 4.6's `instruction_decay` is bimodal — a per-trajectory coinflip between full compliance and total abandonment, taken on the agent's first text turn and stable thereafter — left the mechanism open. Four primer conditions × 10 reps × N=10 on the counter rule (40 Sonnet trajectories, 40 Haiku trajectories) locate it *at N=10*. §3.6 below shows the same cure has no effect at N=100; §3.7 shows it produces opposite-sign effects on a different rule.

**Sonnet 4.6, counter rule, N=10, 10 reps per condition** (baseline reps drawn from `pilot_counter*`, n=5):

| Condition | Pre-task messages | Pattern | Mean | Perfect | Partial | Abandoned |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | (none) | bimodal | 0.345 | 1/5 | 0/5 | 4/5 |
| Zero | 1 user msg, no assistant turn | tri-modal | 0.755 | 3/10 | 6/10 | 1/10 |
| Minimal | 1 trivial exchange (`Hi.` / `ok.`) | bimodal | 0.836 | 8/10 | 0/10 | 2/10 |
| Full | 2 trivial exchanges, no format in priors | always commit | 1.000 | 10/10 | 0/10 | 0/10 |

Two prior trivial off-topic exchanges with no STATUS prefix in the assistant primer responses drive Sonnet 4.6's N=10 compliance from 4/5 abandon to 10/10 perfect. A parallel "format primer" condition (the same two exchanges but with `STATUS: 1` / `STATUS: 2` prefixes on the assistant primer responses) also produced 10/10. The strong "agent commits based on seeing the format applied in prior turns" hypothesis is therefore falsified — format-in-priors is not the load-bearing piece.

What *is* load-bearing decomposes into two effects:

**Any prior assistant turn flips commit-vs-mix.** The minimal primer adds one trivial assistant turn (`ok.`) and the shape snaps back to the same bimodal commit-or-abandon as baseline, with the commit rate elevated. The zero primer — extra user message only, no assistant turn anywhere in the visible history — produces a qualitatively different *tri-modal* shape: three of ten reps commit fully, six produce partial scores in [0.4, 0.95], one fully abandons. The partial reps split as three at 0.909 (drop one turn) and three at 0.545 — the exact categorical-alternation signature observed in discovery-mode pilots (§3.3), applied here to a sequential task. The presence of any prior assistant turn appears to be the discriminator between "clean bimodal commit/abandon" and "noisy mixed compliance".

**Turn count drives commit probability within bimodal mode.** One prior assistant turn: 8/10 commit. Two prior assistant turns: 10/10 commit. The dose-response is monotone over the {0, 1, 2}-turn axis we tested.

**Haiku 4.5, counter rule, N=10, 10 reps per condition:**

| Condition | Mean | Perfect | Abandoned |
| --- | --- | --- | --- |
| Baseline | 1.000 | 10/10 | 0/10 |
| Zero | 1.000 | 10/10 | 0/10 |
| Minimal | 1.000 | 10/10 | 0/10 |
| Full | 1.000 | 10/10 | 0/10 |

Haiku 4.5 was previously tested only on the marker rule (`[[END]]`), where it scored 0.000 across nine trajectories (§3.2). On the counter rule it scores 1.000 across forty trajectories at every primer condition including baseline. The Sonnet warm-up dynamic does not generalize to Haiku here because Haiku has no baseline failure mode on this rule to fix. The same model that universally drops one format rule universally complies with another — rule design has more leverage than priming for Haiku, while priming has more leverage than rule design for Sonnet (which struggles on both rule families).

The cleanest summary, *for N=10 on the counter rule*: bimodal abandonment is a cold-start effect. The agent's first assistant turn is disproportionately likely to drop a format rule that every subsequent turn in the same trajectory would have respected. Inserting any prior assistant turn into the context — trivial, off-topic, format-free — moves the decision-point off the real task and into a context the agent treats as safe to default-comply. The two follow-up subsections narrow this claim substantially.

### 3.6 The N=10 cure does not rescue the long-N drift

The cure characterised in §3.5 works on Sonnet 4.6's counter rule at N=10. Whether it also works at N=100 and N=200 — where atlas baseline compliance is 0.508 and 0.163 respectively (§3.2) — was the obvious next experiment. 10 reps at N=100 plus 10 reps at N=200, all using the full two-trivial-exchange primer that drove N=10 compliance from 0.58 to 1.00:

| Condition | N | Mean | Perfect | Partial | Abandoned |
| --- | --- | --- | --- | --- | --- |
| Atlas baseline | 100 | 0.508 | 4/10 | 2/10 | 4/10 |
| **Full primer** | **100** | **0.470** | **4/10** | **2/10** | **4/10** |
| Atlas baseline | 200 | 0.163 | 1/10 | 1/10 | 8/10 |
| **Full primer** | **200** | **0.011** | **0/10** | **0/10** | **10/10** |

The N=100 distributions are essentially identical with and without the primer — same number of perfect runs, same number of abandons, means within noise. The intervention has *no measurable effect* at N=100. At N=200 every one of 10 primed reps abandoned (mean 0.011), against the atlas baseline of 8/10 abandons (mean 0.163). Under a Bernoulli(0.8) null draw, 10/10 abandons would occur ~11% of the time, so we cannot strictly distinguish "primer hurts" from "primer is null and we drew unluckily." The defensible reading is: the cure does not help at N=200, and the directional effect is opposite to its effect at N=10.

**Implication.** The early-uncertainty dip (N≤10) and the long-context drift (N≥50) are *different* failure modes on Sonnet 4.6's counter rule. The cold-start cure is N=10-specific. The long-N drift needs its own intervention — most plausibly something targeting system-prompt attention as context length grows: periodic rule restatement in user turns, system-prompt-as-tool-result reinjection, or context compaction. None of those have been tested.

### 3.7 The cure does not generalise cleanly across rules either

A second extension: does the cure work on a different Sonnet rule that *also* shows baseline bimodal abandonment? The marker rule from earlier pilot phase (front-loaded `[[END]]`) is the candidate — Sonnet 4.6 abandons it bimodally too (mean 0.79 at baseline N=10, 7/10 perfect / 1/10 partial / 2/10 abandoned), milder than the counter rule but real. Running the same four primer conditions on the marker rule:

| Condition | Marker rule mean | Marker P/Pa/Ab | Counter rule mean | Counter P/Pa/Ab |
| --- | --- | --- | --- | --- |
| baseline | 0.791 | 7 / 1 / 2 | 0.345 | 1 / 0 / 4 |
| zero | **0.582** | 4 / 2 / 4 | **0.755** | 3 / 6 / 1 |
| minimal | 0.791 | 7 / 1 / 2 | 0.836 | 8 / 0 / 2 |
| full | 0.973 | 8 / 2 / 0 | 1.000 | 10 / 0 / 0 |

(10 reps per cell. P/Pa/Ab = perfect / partial / abandoned.)

The dose-response is qualitatively different across rules:

- **Counter rule:** monotone increasing (0.35 → 0.76 → 0.84 → 1.00). Each additional unit of primer content helps.
- **Marker rule:** non-monotone (0.79 → 0.58 → 0.79 → 0.97). The zero-content primer *worsens* compliance; the minimal primer returns to baseline; only the full primer helps.

The zero-content primer shows opposite signs across rules — same model, same N, same intervention, opposite direction. Whatever the agent does on its first text turn when there's a hanging unanswered user message is rule-shape-dependent in a way the four-condition design does not fully resolve. What survives across both rules: **the full two-exchange primer either matches or substantially improves baseline.**

**Implication.** The cold-start dynamic is not a uniform property of "Sonnet 4.6's first-turn instability." The exact decision the agent makes about a system-prompt rule on its first text turn interacts with the rule's specific shape, and the same intervention helps one rule while sometimes hurting another. The robustly-helpful intervention is the full two-trivial-exchange primer; the more degenerate primers have rule-specific effects we do not yet have a clean story for.

### 3.8 Cross-model extension: gpt-3.5-turbo also doesn't show gradient decay

The frontier-model atlas (§3.1-3.4) found zero gradient decay on five models. A natural objection: gradient decay might appear on a weaker model class, where the framework's headline scalar would have a non-vacuous use case. We ran gpt-3.5-turbo at N={5, 10, 25, 50} with 10 reps per cell (N=100 hit an OpenAI quota error mid-run, 40 trajectories total).

| Probe | N=5 | N=10 | N=25 | N=50 | Classified shape |
| --- | --- | --- | --- | --- | --- |
| state_amnesia | 1.000 | 1.000 | 1.000 | 1.000 | perfect |
| instruction_decay | 0.833 | 0.818 | 0.770 | 0.877 | categorical (not gradient) |
| tool_repetition | 1.000 | 1.000 | 0.981 | 0.985 | near-perfect |
| narration_substitution | 1.000 | 1.000 | 1.000 | 1.000 | perfect |

`instruction_decay` is non-monotone over N. Every N=5 rep scored exactly 0.833 (= 5/6 turns compliant) — the "fail first turn then comply" categorical fingerprint observed in Sonnet pilots. At higher N a small fraction of reps fully abandon (1-2/10) and the rest stay near 1.0. Same bimodal-with-categorical-first-turn shape Sonnet shows, just with mildly worse overall compliance and a non-zero tool-repetition rate (0.98 vs frontier-model 1.000). **No gradient decay on gpt-3.5-turbo either.** The negative finding extends to a meaningfully weaker model class than the five we originally tested.

### 3.9 `premature_termination` probe and `find_max` task

The pre-registration listed five probes; v0.1.0 shipped four because `find_and_synthesise` forecloses `premature_termination` by enumerating the work upfront. The fifth probe is now shipped, along with a new task (`find_max`) designed to admit it: agent gets N topics with hidden integer values, looks up each via `lookup(topic)`, and submits the maximum. The system prompt grants explicit licence to be efficient ("don't make more calls than needed"). Agents that take the licence submit before exhausting lookups; the probe scores `n_completed / n_required`.

Sonnet 4.6 at N=10 and N=30:

| N | Sonnet 4.6 n_completed (median) | premature_termination |
| --- | --- | --- |
| 10 | 10 of 10 (every rep) | 1.000 (10/10 reps) |
| 30 | 30 of 30 (every rep) | 1.000 (10/10 reps) |

Sonnet 4.6 does not take the bait. Even at N=30 it exhausts every lookup, finds the true maximum, and never exhibits premature termination. The probe and task have full unit-test coverage; they just do not fire on this model on this task. Eliciting the failure mode reliably would likely require a weaker model (where thoroughness is less default) or a task with truly open-ended completion criteria (no finite topic list to enumerate).

### 3.10 What rescues the long-N drift: strategy lock-in pre-empted by initial-message reinforcement

§3.6 left the long-N failure mode unresolved: the cold-start cure does not work at N=100 or N=200, so something other than cold-start uncertainty is driving the bimodal abandonment at those depths. A three-experiment progression (~$13.50, 39 trajectories) locates the mechanism and produces a robust cure for prefix-style rules.

**Experiment 1 — periodic mid-trajectory reminders.** The natural first hypothesis was system-prompt attention fade: re-surface the rule in user-side content every K turns and compliance should hold. 9 reps at N=100 with a 30-token rule reminder appended to every K=10-th tool_result user message (10 reminders evenly spaced through 100 turns):

| Condition | N | Mean | Perfect | Partial | Abandoned |
| --- | --- | --- | --- | --- | --- |
| Baseline | 100 | 0.508 | 4/10 | 2/10 | 4/10 |
| Periodic K=10 reminder | 100 | 0.701 | 5/9 | 2/9 | 2/9 |

The mean lifts but the bimodal failure mode persists. Inspecting per-rep compliance maps reveals the actual structure: every one of the 9 reps falls into one of exactly three discrete behaviours — `1111111111|1111111111|...` (perfect, 5 reps), `1.1.1.1.1.|1.1.1.1.1.|...` (stable alternation insensitive to reminders, 2 reps), or `1.1.1.1.1.|1.........|1.........|...` (one-shot rescue: comply on the turn right after each reminder, ignore for the next 9, 2 reps). The aggregate "fade curve" — d=0 right-after-reminder = 100% (90/90), declining to ~56% at d=1 — is an artifact of mixing these three discrete strategies, not graded attention decay. **The attention-fade hypothesis is falsified**: a fading agent would show graded compliance across d=0..9 within trajectories; instead the data shows a step function reflecting categorical strategies.

The diagnostic detail: both abandon-class reps showed `1.1.1.1.1.` alternating compliance for the first 10 turns, *before* the first reminder fires. Strategy commitment appears to happen in the opening turns; mid-trajectory reminders only briefly perturb an already-committed strategy.

**Experiment 2 — reminder visible from turn 1.** If strategy lock-in happens early, the rescue is to deliver rule reinforcement before the agent has a chance to commit. 10 reps at N=100, same reminder text appended to the initial user message (so the agent sees it before producing turn 1), then K=10 cadence thereafter:

| Condition | N | Mean | Perfect | Partial | Abandoned |
| --- | --- | --- | --- | --- | --- |
| Baseline | 100 | 0.508 | 4/10 | 2/10 | 4/10 |
| Periodic K=10 only | 100 | 0.701 | 5/9 | 2/9 | 2/9 |
| **Reminder from turn 1 + K=10** | **100** | **1.000** | **10/10** | **0/10** | **0/10** |

Every one of 1010 assistant turns complied. The intervention is binary: pre-empting strategy commitment with a reminder at turn 1 takes the bimodal distribution from coinflip to delta-at-1.

**Experiment 3 — depth verification at N=200.** Baseline at N=200 is mean 0.163. If the cure is real and not a fragile N=100 artifact, it should hold at twice the depth:

| Condition | N | Mean | Perfect | Partial | Abandoned |
| --- | --- | --- | --- | --- | --- |
| Baseline | 200 | 0.163 | 1/10 | 1/10 | 8/10 |
| **Reminder from turn 1 + K=10** | **200** | **1.000** | **10/10** | **0/10** | **0/10** |

10/10 perfect across 2010 assistant turns, against a 6x lower baseline. The cure has no depth ceiling within the tested range.

**Experiment 4 — marker rule generality.** §3.7 showed the cold-start cure was rule-shape-dependent. The obvious risk for the new intervention is rule-specific success. 10 reps at N=100 on the marker rule with the same reminder-from-start protocol:

| Condition | Rule | N | Mean | Perfect | Partial | Abandoned |
| --- | --- | --- | --- | --- | --- | --- |
| Reminder from turn 1 + K=10 | counter | 100 | 1.000 | 10/10 | 0/10 | 0/10 |
| **Reminder from turn 1 + K=10** | **marker** | **100** | **0.191** | **0/10** | **2/10** | **8/10** |

The intervention that produced flawless compliance on the counter rule produces near-complete abandonment on the marker rule. The pattern is also qualitatively different: every rep shows zero compliance for turns 1-10 — the initial-message reminder accomplished nothing for the first 10 turns of any rep. Of the 10 reps, 2 then snap to full compliance the moment the first mid-trajectory reminder lands at turn 11 and sustain perfectly through turn 100 (score 0.901), 1 falls into the one-shot rescue pattern (score 0.099), and 7 never apply `[[END]]` at all.

**Implication.** The mechanism driving long-N bimodal abandonment on Sonnet 4.6's counter rule is **early-trajectory strategy commitment**, pre-emptable by restating the rule in the initial user message. What's now ruled out: system-prompt attention fade, accumulating tool-result token dilution, and any continuous-decay framing. The cure is depth-robust within the tested range (10/10 at N=100 AND N=200) but rule-shape-dependent (10/10 on counter, 0/10 on marker). The asymmetry between prefix/stateful rules and suffix/stateless rules is itself a finding — Sonnet's commitment dynamics interact with where in the response the rule expects compliance. Plausible interpretations (not data-supported, just hypotheses): token-position attention bias (prefix slots in at the most-attended position), `[[END]]`-as-stop-token semantic confusability, or the initial-message reminder being read as a clarification of the current request rather than a persistent behavioural rule.

## 4. Discussion

### 4.1 The halftrace concept needs reframing for modern frontier models

The pre-registered framework assumed that compliance gradually degrades with trajectory length. We did not observe gradient decay in any of sixteen pilots, on either model, across any probe or task variant. We did observe two non-gradient shapes — bimodal per trajectory and categorical per turn-type — that the halftrace scalar is undefined for, plus a third tri-modal shape that emerges under a specific cold-start condition (§3.5).

The reframed [halftrace v0.1.0](https://github.com/ruairidhwm/halftrace) classifies compliance into six shapes (perfect, abandoned, bimodal, categorical, gradient, unclassified) and emits `commit_probability` as the headline metric, with `halftrace` only populated when shape is `gradient`. On the pilot data, that final field is never populated.

### 4.2 Compliance is a categorical decision, not a continuous variable

The cleanest empirical claim from the pilot phase is structural: Sonnet 4.6 makes compliance decisions *per rule-class* rather than per *amount of context*. In sequential mode, the decision is per trajectory (commit or abandon on turn 4, stable thereafter). In discovery mode, the decision is per turn-type (always comply on planning turns, never on acknowledgement turns). N influences neither.

This is genuinely useful information for agent developers: it predicts that prompt-engineering interventions which affect rule engagement on turn 4 (worked examples, clearer scope statements) will move commit_probability a lot, while interventions targeting "the rule must hold at long N" will move it not at all on the current model class.

### 4.3 The N=10 decision-point is the first assistant turn — but there are at least two decision-points

The mechanism investigation at N=10 (§3.5) sharpens "categorical commit decision" into a more specific claim *at that trajectory length*: the relevant decision is taken on the agent's first assistant turn, and the probability of a commit outcome is set by what message history exists before that turn. Two prior trivial assistant turns collapse the distribution to a delta at compliance = 1. One prior assistant turn moves commit probability from ~0.2 to ~0.8. Zero prior assistant turns (with one prior user turn) yield a tri-modal mixing of commit, alternate, and abandon — for the counter rule.

§3.6 and §3.7 then narrow this claim along two axes. **The same intervention at N=100 has no measurable effect** — the cure does not transfer to the long-context drift regime, which is therefore a *separate* failure mode with a different mechanism. **The same intervention on the marker rule produces a non-monotone dose-response**, with the zero-content primer actually worsening compliance; only the full two-exchange primer is robustly helpful.

§3.10 then closes the long-N question and substantially sharpens the mechanism story. The long-N failure mode is **not attention fade** — periodic mid-trajectory reminders produce a step-function compliance pattern (three discrete behaviours: perfect, stable alternation insensitive to reminders, and one-shot rescue) rather than the graded compliance a fading mechanism would produce. The actual mechanism is **early-trajectory strategy commitment**: the agent picks a compliance strategy in the first ~10 turns and locks in, with mid-trajectory perturbations only briefly affecting an already-committed strategy. The cure is to pre-empt the commitment by restating the rule in the initial user message — which on the counter rule drives compliance to 10/10 perfect at both N=100 *and* N=200 (2010 perfect assistant turns at a baseline mean of 0.16 for N=200). The cure is depth-robust within tested range but **rule-shape-dependent**: the same intervention produces 0/10 perfect on the marker rule at N=100, with all reps showing zero compliance for the first 10 turns despite the reminder being visible at turn 1.

The claims that now survive, after all three narrowing rounds:

- **Sonnet 4.6's bimodal abandonment on a prefix/stateful rule has a single mechanism across the full N range we tested**: early-trajectory strategy commitment by turn ~10. The N=10 cold-start dip is the short-N face of this mechanism; the N=100/200 drift is the long-N face. Both are curable with interventions that move rule reinforcement into the visible context before commitment forms — a pre-task warm-up primer at short N, or rule restatement in the initial user message at any N.
- **For prefix rules, the most robust intervention we have measured is rule restatement in the initial user message.** It works at both N=100 and N=200 (10/10 perfect at each, against baseline means of 0.51 and 0.16). It supersedes the warm-up primer for any trajectory length ≳ 25.
- **For suffix rules, neither known intervention reliably rescues at long N.** On the marker rule at N=100, even the initial-message reminder fails to budge the first 10 turns; 7/10 reps then never apply the rule at all. Why prefix and suffix rules engage different commitment dynamics on Sonnet is open.
- **The cleanest falsified hypotheses are attention fade and tool-result-token dilution.** The periodic-reminder data shows compliance is mechanically determined by the early-turn strategy choice plus discrete one-shot rescues, not by graded loss of rule salience as context grows.

The theoretical implication: "context rot" framings that look at late-N aggregate compliance miss the early commitment point that mechanistically *causes* the late-N pattern. The aggregate compliance curve at long N is not measuring how attention decays with context length; it is measuring how often the agent commits to a non-compliant strategy in the opening turns. The intervention that moves the curve is intervention on the opening turns.

### 4.4 Task design is the actual bottleneck

Three of four probes scored 1.0 universally on `find_and_synthesise`. The task gives the agent a complete, well-specified enumeration of work with serial execution. There is no opportunity for the agent to repeat, substitute narration for action, or terminate prematurely. Probes that target these failure modes can only fire on tasks that admit them. Building such tasks is the largest open problem; halftrace v0.1.0 does not solve it.

## 5. Limitations

- **Only Anthropic models tested.** Sonnet 4.6 and Haiku 4.5. The negative finding may not extend to GPT, Llama, Gemini, or smaller open models, which is precisely the population where gradient decay is most likely to appear. An OpenAI adapter is a clear extension.
- **One task family.** All pilots used `find_and_synthesise` or its variants. A genuinely adversarial task family (open-ended search, multi-document reasoning, conflicting constraints) might elicit any of the foreclosed failure modes.
- **2-3 reps per cell on the original probe pilots; 10 reps per cell on the mechanism conditions.** The pre-registered design called for 5. Bimodal probes need 7-10 to estimate the commit-rate stably; 3 can produce spurious halftrace estimates that vanish with more samples. The mechanism cells reach this threshold; the broader N-axis pilots do not.
- **The API is not bit-deterministic.** Even at low effort settings, repeated calls with the same prompt and seed produced different responses (one rep scored 0.182, another 1.000 with identical configuration). Within-pilot deterministic patterns we initially observed reflected lucky draws, not stable inference behaviour.
- **The mechanism findings are sharp but still one model and two rules.** The strategy-lock-in mechanism and its initial-message-reminder cure are characterised on Sonnet 4.6 with the counter rule, and replicate at two depths (10/10 perfect at both N=100 and N=200 — 2010 perfect assistant turns). On Sonnet's marker rule the cure fails (0/10 perfect at N=100). Haiku 4.5 has no baseline failure on the counter rule, so the mechanism replication is vacuous there. gpt-3.5-turbo also does not show gradient decay (§3.8). What remains untested: a third Sonnet rule (content rule, citation rule, tone rule) to determine whether the prefix/suffix asymmetry generalises along the structural axis we're guessing at, or whether it reflects something rule-specific we haven't yet decomposed; whether other "perfect" frontier models (Opus, GPT-4.1, GPT-4o) would exhibit comparable strategy-lock-in dynamics on a rule that did produce baseline failure for them; and whether the cure holds at N > 200, where commitment dynamics might eventually be overtaken by a separately-emerging mechanism we have not yet detected.

## 6. Conclusion

A pre-registered study aiming to measure halftraces for four failure modes on modern Claude produced zero halftraces in any configuration across ~230 atlas trajectories on five frontier models plus 40 cross-model trajectories on gpt-3.5-turbo. The negative result is itself the contribution: it pushes back on the "context rot" intuition with concrete evidence and reframes the diagnostic primitive from *decay rate* to *compliance shape*. A multi-stage mechanism investigation then characterised the bimodal abandonment on Sonnet 4.6 across both the trajectory-length and rule-design axes: a cold-start cure (two-trivial-exchange primer) at short N (§3.5), its failure at long N (§3.6), its rule-shape-dependence (§3.7), and finally — after a periodic-reminder experiment that falsified the attention-fade hypothesis by showing step-function rather than graded compliance — a long-N rescue (§3.10) in which restating the rule in the initial user message drives compliance from baseline mean 0.51 / 0.16 to 10/10 perfect at N=100 *and* N=200 on the counter rule (2010 perfect assistant turns). The same intervention fails on the marker rule (0/10 perfect at N=100), so the cure is depth-robust but rule-shape-dependent. What survives, sharpened: the mechanism driving Sonnet's long-N bimodal abandonment on prefix/stateful rules is early-trajectory strategy commitment by turn ~10; it is pre-emptable by moving rule reinforcement into the visible context before commitment forms; and "context rot" framings that look at late-N aggregate compliance miss the early commitment point that mechanistically causes the late-N pattern. The reframed tool (halftrace, now shipping `premature_termination` + `find_max` + LangSmith ingest) ingests existing OpenAI / Anthropic / LangSmith logs, classifies the shape of compliance per probe, and emits commit-probability plus per-shape diagnostic suggestions — including the validated initial-message-reminder cure for bimodal abandonment, with the marker-rule limitation documented. The original halftrace scalar still works when shape is gradient; the pilot data — across two model families, six models, three rule designs, two task variants, and N from 5 to 200 — simply never produced that shape.

## Acknowledgements

This is a one-person project written with substantial assistance from Claude Opus 4.7 (1M context) operating in Claude Code. The agentic build-and-pilot loop that produced both the tool and these findings is itself an artefact worth noting.

## References

- Companion repo with code, raw pilot data, and reproducible pipeline: <https://github.com/ruairidhwm/halftrace>
- Original pre-registration: [`HYPOTHESES.md`](../HYPOTHESES.md)
- Detailed pilot ledger and findings: [`RESULTS.md`](../RESULTS.md)
