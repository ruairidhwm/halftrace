"""Actionable diagnostics for compliance profiles.

A `ComplianceProfile` tells you *what shape* of compliance your agent has.
A `Diagnosis` tells you *why* and *what to try next*. The catalog of
shape-to-cause-to-suggestion mappings is grounded in the pilot phase
findings (RESULTS.md) rather than generic prompt-engineering advice:
each entry refers to a pattern we actually observed in the data.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from halftrace.fit import ComplianceProfile, Shape


class Diagnosis(BaseModel):
    """Human-readable cause + suggestions for a compliance shape."""

    model_config = ConfigDict(extra="forbid")

    shape: Shape
    cause: str
    suggestions: list[str]


_DIAGNOSES: dict[Shape, tuple[str, list[str]]] = {
    "perfect": (
        "Agent reliably follows the rule across every trajectory at every N value tested.",
        ["No action needed."],
    ),
    "abandoned": (
        "Agent ignores the rule from turn 1 across every trajectory. "
        "In the pilot phase this was observed when the rule was buried in the system "
        "prompt or stated too abstractly for the model to operationalise.",
        [
            "Move the rule to the first line of your system prompt.",
            "Add a concrete worked example showing a correctly-formatted response.",
            "Verify the model is capable of the rule at all — start with a simpler "
            "variant (literal marker rather than computed counter, etc.) to isolate "
            "whether the issue is format or content.",
        ],
    ),
    "bimodal": (
        "Agent commits-or-abandons per trajectory: the choice is approximately a "
        "coinflip and is stable for the rest of the trajectory once made. In the "
        "pilot phase the first text turn (before any tool result has come back) "
        "almost always violated the rule; after the first tool-result round trip "
        "the agent either locked into compliance for the remainder or never engaged. "
        "This is the most fixable shape with prompt engineering.",
        [
            "Add a worked example response in the system prompt. Agents that see "
            "the rule modelled tend to lock into compliance.",
            "Restate the rule in the initial user message in addition to the system "
            "prompt — agents are more likely to follow rules they've seen recently.",
            "Consider relaxing the rule on turn 1 (before any tool result). The "
            "first text turn is consistently the weakest compliance point.",
        ],
    ),
    "categorical": (
        "Agent applies the rule to one turn-type and drops it on another, "
        "stably across N. The compliance fraction is determined by the "
        "ratio of turn-types in the trajectory, not by context length. In the "
        "pilot phase this manifested as an exact alternating bit pattern: "
        "the rule was applied on discovery/planning turns and dropped on "
        "lookup/acknowledgement turns even though the rule was stated universally.",
        [
            "State the rule's scope explicitly. Replace 'every response' with "
            "'EVERY assistant message that contains any text, including tool "
            "acknowledgments and one-line intermediate narration'.",
            "Inspect details.per_turn on the probe's Score output to identify "
            "which turn-type the agent is excluding.",
            "Consider whether the categorical pattern is acceptable for your "
            "use case. It is stable, predictable, and a real-world agent "
            "developer may not need 100% compliance.",
        ],
    ),
    "gradient": (
        "Compliance smoothly degrades as N grows — the classical 'context rot' "
        "shape. Across the pilot phase this was not observed on modern frontier "
        "Claude (Sonnet 4.6, Haiku 4.5) at any tested N up to 200. If you are "
        "seeing it, the model is likely smaller or older than current frontier "
        "Claude, or the task imposes a load that approaches a real capability ceiling.",
        [
            "Reduce per-turn context: shorter tool results, more concise prompts, "
            "less metadata threaded through each round.",
            "Use context-editing or compaction features to clear stale turns "
            "without breaking the agent's working state.",
            "Restate the rule periodically in the user turn at the depth where "
            "compliance starts dropping (the halftrace value indicates roughly where).",
        ],
    ),
    "unclassified": (
        "Compliance pattern doesn't fit any of the canonical shapes. The N range "
        "may be too narrow, the rep count too low, or the rule may be eliciting "
        "a behaviour we haven't catalogued. In the pilot phase this fired when "
        "two reps at the same cell produced very different scores (one categorical, "
        "one perfect) — the shape classifier refuses to claim a pattern that "
        "isn't well-supported by the data.",
        [
            "Increase reps per N. Bimodal patterns can look unclassified at low "
            "sample counts; the pilot phase used 3 reps and recommends 5-10 for "
            "any probe showing rep-variance above 0.3 within a cell.",
            "Widen the N range. A pattern that looks unclassified across {5, 10, "
            "25} may resolve into gradient or bimodal at {25, 50, 100, 200}.",
            "Inspect details.per_turn on the underlying Score objects to see what "
            "the agent is actually doing turn-by-turn.",
        ],
    ),
}


def diagnose(profile: ComplianceProfile) -> Diagnosis:
    """Return a Diagnosis with a likely cause and concrete suggestions.

    The cause/suggestions catalog is grounded in the pilot-phase findings;
    see RESULTS.md for the supporting evidence.
    """
    cause, suggestions = _DIAGNOSES[profile.shape]
    return Diagnosis(shape=profile.shape, cause=cause, suggestions=list(suggestions))
