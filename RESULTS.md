# Pilot results

Pre-pilot status as of 2026-05-14. The pilots described here are scoped
to validate the halftrace measurement pipeline against real model
behaviour at small N, before committing to the full 120-trajectory study
laid out in `HYPOTHESES.md`. They are not the full study.

## Summary

Seven pilot runs against the Anthropic API (~39 trajectories, $5.83 of
spend) end-to-end validated the pipeline: `Task` → adapter → `Trajectory`
→ probe → `fit_halftrace` → CLI. Two probes are implemented
(`state_amnesia`, `instruction_decay`) and have been measured on two
models (`claude-sonnet-4-6`, `claude-haiku-4-5`).

**Neither probe produced a halftrace in the tested N range.** That is
itself the finding worth recording — the pilots surfaced concrete design
issues that need to be resolved before the full study is worth running.

## Pipeline status

| Component | Status |
| --- | --- |
| `Trajectory` schema + JSONL serialization | Done |
| `Probe` protocol + `Score` type | Done |
| `state_amnesia` probe | Done |
| `instruction_decay` probe | Done |
| `tool_repetition`, `narration_substitution`, `premature_termination` | Not yet implemented |
| `fit_halftrace` (linear interp + bootstrap 95% CI) | Done |
| Anthropic adapter (manual tool-use loop, caching, usage tracking) | Done |
| OpenAI adapter | Not yet implemented |
| `find_and_synthesise` task | Done (with `--serial` and marker-rule support) |
| Pilot runner (`halftrace` CLI) | Done — multi-probe, JSONL output, per-probe halftrace fitting |

## Findings

### `state_amnesia`

| Model | N=5 | N=10 | N=25 | N=50 | N=100 | N=200 |
| --- | --- | --- | --- | --- | --- | --- |
| `claude-sonnet-4-6` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `claude-haiku-4-5` | 1.000 | 1.000 | 1.000 | — | — | — |

All cells scored 1.0; no decay observed. The probe is mechanically
sound — it correctly captured a single planted codeword across **400+
intervening trajectory turns** at N=200 under forced serialization.

**Implication.** H1's prediction that `state_amnesia` halftrace is above
N=120 holds on Sonnet 4.6 with at least N > 200. The probe in its
current form does not separate modern Claude models on single-fact
recall in this trajectory length range. To produce a halftrace at the
intended N axis the task needs more demanding retention: multiple
concurrent plants, distractor information, semantic interference, or
recall under load.

### `instruction_decay`

Marker rule `[[END]]` at the end of every assistant text response, with
the rule front-loaded in the system prompt and an explicit format
example. Per-rep scores, not just per-N means:

| Model | N=5 | N=10 | N=25 |
| --- | --- | --- | --- |
| `claude-sonnet-4-6` | 0.833, 0.167, 0.167 | 0.091, 0.000, 0.818 | 0.962, 0.962, 0.962 |
| `claude-haiku-4-5` | 0.000, 0.000, 0.000 | 0.000, 0.000, 0.000 | 0.000, 0.000, 0.000 |

Two distinct patterns:

1. **Sonnet 4.6: bimodal lock-in.** Every trajectory's first text turn
   (before any tool result has come back) violates the rule. After the
   first tool-result round trip, the agent either picks the rule up and
   complies for the remainder of the trajectory, or it never engages
   with the rule at all. Per-rep variance is large at small N (a
   coinflip per trajectory); at N=25 most reps lock in early enough that
   the mean is 0.962.
2. **Haiku 4.5: total non-compliance.** Across every rep at every N,
   Haiku ignored the marker rule entirely.

Neither pattern is a halftrace shape. The probe correctly captures
instruction adherence, but the chosen rule produces a binary
"engage / don't engage" outcome rather than a smooth degradation curve.

**Implication.** A useful `instruction_decay` rule needs (a) initial
compliance near 1.0 across reps, (b) gradual erosion under context
pressure. A marker that the model can ignore from turn 1 satisfies
neither. Candidates worth piloting next: stylistic constraints (e.g.,
respond in third person), counter-attention constraints (e.g., always
verify N before answering), or content-injection constraints that
require the agent to thread a specific phrase through every turn.

## Methodological learnings

These are concrete design lessons from the pilot phase that should
shape the full study.

### Parallel tool use defeats trajectory-length scaling

The first pilot (`results/pilot.jsonl`) revealed that Sonnet 4.6 emits
all N `lookup` calls as a single parallel `tool_use` response. Plant and
recall annotations end up adjacent in the same user message, so the
"N tool calls" axis does not produce N rounds of context accumulation.
Adding `tool_choice: {type: "auto", disable_parallel_tool_use: true}`
(exposed via `--serial`) forces one tool call per assistant turn and
restores genuine sequential context depth. **The full study must run
with `--serial` or an equivalent control.**

### Prompt caching is essential at scale

The deep pilot at N=200 cost $4.03 for 9 trajectories. Of 7.4M total
input tokens, 7.2M were cache reads (10% of base price). Without prefix
caching, the same pilot would have cost an order of magnitude more.
Haiku's prefix at N≤25 is below its 4096-token cache threshold, so the
Haiku pilot did not benefit — this is worth verifying before extending
Haiku to larger N.

### Three reps per cell is too few for `instruction_decay`

The Sonnet 4.6 `instruction_decay` data shows rep-to-rep variance that
swamps any N-trend at three reps per cell. The pre-registered design
calls for 5 reps; even that may be inadequate for bimodal probes. A
post-pilot recommendation: bump to 7–10 reps per cell for any probe
where rep variance > 0.3 within a single cell.

### "Fail-then-comply" is a distinct failure mode

The N=25 `instruction_decay` trajectories that scored 0.962 all share
the same shape: turn 2 (the agent's first text turn, before any tool
output) violates the rule; every subsequent text turn complies. This
suggests modern Claude models read the system prompt instruction but
defer rule-engagement until they have at least one round of grounded
interaction. This is its own measurable phenomenon and may deserve its
own probe distinct from "instruction decay".

## Cost ledger

| Pilot | Spec | Cost |
| --- | --- | --- |
| Smoke | Sonnet 4.6, N=5, reps=1, parallel | $0.012 |
| First | Sonnet 4.6, N={5,10,25}, reps=3, parallel | $0.180 |
| Serial | Sonnet 4.6, N={5,10,25}, reps=3, serial | $0.440 |
| Deep | Sonnet 4.6, N={50,100,200}, reps=3, serial | $4.030 |
| instr-v1 | Sonnet 4.6, N={5,10,25}, reps=3, marker rule buried | $0.460 |
| instr-v2 | Sonnet 4.6, N={5,10,25}, reps=3, marker rule front-loaded | $0.450 |
| haiku | Haiku 4.5, N={5,10,25}, reps=3, marker rule front-loaded | $0.250 |
| **Total** | | **$5.82** |

Well under the £200 cap from `HYPOTHESES.md` § Stopping conditions. The
deep pilot at N=200 is the most relevant per-trajectory cost reference
for the full study: ~$0.45 per N=200 trajectory at Sonnet 4.6 with
caching.

## What's needed before the full study runs

1. **Probe redesign so at least one of the four failure modes actually
   produces a halftrace at the intended N axis.** Either harden the
   tasks (concurrent plants, distractor load) so `state_amnesia` decays
   in the {5..200} range, or pick `instruction_decay` rules that
   produce smooth curves rather than bimodal lock-in.
2. **Implement the remaining three probes** (`tool_repetition`,
   `narration_substitution`, `premature_termination`). They are
   mechanically straightforward — pattern is established by the first
   two — but design questions raised by this pilot apply to them too.
3. **OpenAI adapter** for the GPT-4o / GPT-4.1 arms of the study.
   Mechanically a port of the Anthropic adapter; deferred until probes
   are settled.
4. **Bump reps to 5–10 per cell** for any probe showing rep-variance
   above 0.3 within a cell.
5. **Pre-register the post-pilot probe/task changes** so the eventual
   study results remain interpretable against the original hypotheses.

## What worked first try

For completeness — these components did not surface issues during the
pilot phase:

- Trajectory schema (pydantic, append-only, JSONL round-trip)
- Probe protocol and `Score` model
- `fit_halftrace` linear interpolation + bootstrap CI
- Anthropic adapter manual tool-use loop including `end_turn`
  termination and stop_reason recording
- Prompt caching of system + last tool spec
- CLI argument parsing, dry-run mode, and multi-probe scoring
