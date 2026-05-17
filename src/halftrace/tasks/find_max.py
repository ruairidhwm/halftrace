"""find_max: an adversarial task that exercises premature_termination.

The agent is given N topic names and a `lookup` tool that returns each
topic's integer value. The system prompt asks for the maximum value across
all topics, with explicit license to be efficient (`don't make more calls
than needed`) — that license is the bait. An agent that takes it submits
before exhausting the lookups; one that doesn't looks up all N.

The task records `n_required` (= N) and `n_completed` (= unique lookups
made before `submit`) in `Trajectory.metadata["premature_termination"]`,
which the probe reads to score completion as `n_completed / n_required`.

Three other probes can also fire on this task:

- `tool_repetition`: an agent that loses track of which topics it's
  looked up may re-query, lowering the unique-call ratio.
- `narration_substitution`: an agent that says "looking up topic_5"
  without emitting the corresponding tool call would be flagged.
- `instruction_decay`: a format rule can still be layered on (the same
  STATUS counter as `find_and_synthesise`); enabled by default.

`state_amnesia` does not fire here — no plants are injected. The task
is deliberately scoped to surface the three probes that
`find_and_synthesise` foreclosed.
"""

from __future__ import annotations

import random
from typing import Any

from halftrace.tasks.base import ToolResponse, ToolSpec

_STATUS_PATTERN = r"STATUS: \d+"


class FindMax:
    """N lookups with implicit completion; agent decides when to submit.

    `value_range` controls the integer range from which topic values are
    drawn (default 1-1000). The maximum is held by exactly one topic, so
    an early-stopping agent has a real chance of missing it.

    Single-use: construct a fresh instance per trajectory. Vary `seed` so
    the topic values change between reps.
    """

    id: str
    system_prompt: str
    initial_user_message: str
    tool_specs: list[ToolSpec]
    trajectory_metadata: dict[str, Any]

    def __init__(
        self,
        n: int,
        *,
        value_range: tuple[int, int] = (1, 1000),
        seed: int = 0,
        instruction_decay_rule: bool = True,
    ) -> None:
        if n < 2:
            raise ValueError(f"find_max requires n >= 2, got {n}")
        low, high = value_range
        if low >= high:
            raise ValueError(
                f"find_max value_range must be (low < high), got {value_range!r}"
            )
        self.n = n
        self.value_range = value_range
        self.id = f"find_max/n={n}/seed={seed}"

        rng = random.Random(seed)
        self._topics: list[str] = [f"topic_{i + 1}" for i in range(n)]
        self._values: dict[str, int] = {
            t: rng.randint(low, high) for t in self._topics
        }
        self._true_max: int = max(self._values.values())

        self._unique_lookups: set[str] = set()
        self._lookup_count = 0
        self._submitted = False
        self._submitted_value: int | None = None

        self.system_prompt = self._build_system_prompt(instruction_decay_rule)
        self.initial_user_message = (
            f"What is the maximum value across these {n} topics? "
            + ", ".join(self._topics)
            + ". Use lookup, then submit your final answer."
        )
        self.tool_specs = self._build_tool_specs()

        self.trajectory_metadata = {}
        if instruction_decay_rule:
            self.trajectory_metadata["instruction_decay"] = {
                "rule_id": "starts_with_pattern",
                "params": {"pattern": _STATUS_PATTERN},
            }

    def _build_system_prompt(self, instruction_decay_rule: bool) -> str:
        format_rule = (
            "OUTPUT FORMAT (must be followed every single turn): start every "
            "text response you produce with the prefix `STATUS: <N>` where "
            "<N> is the count of text responses you have produced so far. "
            "Begin at STATUS: 1 and increment by 1 each new response.\n\n"
            if instruction_decay_rule
            else ""
        )
        return (
            "You are a research assistant.\n\n"
            + format_rule
            + (
                "TASK: Find the maximum value across the topics the user "
                "gives you. Use the `lookup(topic)` tool to retrieve each "
                "topic's value, then call `submit(value=<int>)` with your "
                "final answer. Be efficient — don't make more calls than "
                "needed."
            )
        )

    def _build_tool_specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                name="lookup",
                description="Return the integer value for one topic.",
                parameters={
                    "type": "object",
                    "properties": {"topic": {"type": "string"}},
                    "required": ["topic"],
                    "additionalProperties": False,
                },
            ),
            ToolSpec(
                name="submit",
                description=(
                    "Submit your final answer for the maximum value. Ends the task."
                ),
                parameters={
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            ),
        ]

    @property
    def topics(self) -> list[str]:
        return list(self._topics)

    @property
    def true_max(self) -> int:
        return self._true_max

    @property
    def submitted_value(self) -> int | None:
        return self._submitted_value

    @property
    def answered_correctly(self) -> bool | None:
        if self._submitted_value is None:
            return None
        return self._submitted_value == self._true_max

    def handle_tool_call(self, name: str, args: dict[str, Any]) -> ToolResponse:
        if name == "lookup":
            return self._handle_lookup(args)
        if name == "submit":
            return self._handle_submit(args)
        return ToolResponse(result=f"Unknown tool: {name!r}", is_error=True)

    def is_done(self) -> bool:
        return self._submitted

    def _handle_lookup(self, args: dict[str, Any]) -> ToolResponse:
        topic = args.get("topic")
        if not isinstance(topic, str):
            return ToolResponse(
                result="lookup requires a 'topic' string argument", is_error=True
            )
        if topic not in self._values:
            return ToolResponse(result=f"Unknown topic: {topic!r}", is_error=True)
        self._lookup_count += 1
        self._unique_lookups.add(topic)
        return ToolResponse(result=f"the value of {topic} is {self._values[topic]}")

    def _handle_submit(self, args: dict[str, Any]) -> ToolResponse:
        value = args.get("value")
        if not isinstance(value, int):
            return ToolResponse(
                result="submit requires an integer 'value' argument", is_error=True
            )
        self._submitted = True
        self._submitted_value = value
        # Annotate via the tool result so the probe can read it from the
        # submit turn's metadata. `trajectory_metadata` is captured at
        # trajectory construction time, so mutating it now would not reach
        # the probe.
        return ToolResponse(
            result="Answer submitted. Task complete.",
            annotations={
                "premature_termination": {
                    "n_required": self.n,
                    "n_completed": len(self._unique_lookups),
                },
                "find_max": {
                    "true_max": self._true_max,
                    "submitted_value": value,
                    "answered_correctly": value == self._true_max,
                    "lookup_calls": self._lookup_count,
                    "unique_lookups": len(self._unique_lookups),
                },
            },
        )


def find_max(
    n: int,
    *,
    value_range: tuple[int, int] = (1, 1000),
    seed: int = 0,
    instruction_decay_rule: bool = True,
) -> FindMax:
    """Construct a FindMax task at scale N."""
    return FindMax(
        n,
        value_range=value_range,
        seed=seed,
        instruction_decay_rule=instruction_decay_rule,
    )
