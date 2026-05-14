"""find_and_synthesise: the canonical agentic task for the context-rot study.

The agent is given N topics and asked to look up each one via the `lookup`
tool, then call `submit_summary` to end the task. `n_plants` codewords are
injected into lookup responses at evenly-spaced positions across the
trajectory; the final lookup asks the agent to recall all of them. The
agent's text response after the recall is what the `state_amnesia` probe
scores (per-codeword, aggregated to a single fraction).
"""

from __future__ import annotations

import random
from typing import Any

from halftrace.tasks.base import ToolResponse, ToolSpec

_CODEWORDS = (
    "HORSE",
    "TIGER",
    "FALCON",
    "OTTER",
    "SALMON",
    "MOOSE",
    "BISON",
    "RAVEN",
    "WOLF",
    "EAGLE",
)

_STATUS_PATTERN = r"STATUS: \d+"


def _plant_id(index: int) -> str:
    return f"password_{index + 1}"


def _plant_positions(n: int, n_plants: int) -> list[int]:
    """Evenly-spaced lookup indices in [0, n-1) for the planted facts.

    Recall fires on lookup index ``n-1``; plants must precede it. Guaranteed
    unique whenever ``n_plants <= n - 1``.
    """
    return [i * (n - 1) // n_plants for i in range(n_plants)]


def _random_codeword(rng: random.Random) -> str:
    return f"{rng.choice(_CODEWORDS)}{rng.randint(10, 99)}"


class FindAndSynthesise:
    """N lookups + a submit, with one or more plants spread across the trajectory.

    Construct directly (`FindAndSynthesise(n=10, n_plants=3)`) or via the
    convenience factory `find_and_synthesise(10, n_plants=3)`. `seed`
    controls the randomised topic facts and the planted codewords for
    reproducibility.

    `n_plants` codewords are injected at evenly-spaced lookup positions in
    `[0, n-1)` (inclusive of 0, exclusive of `n-1` which is reserved for the
    recall). All planted codewords are recalled together in one final
    question; the state_amnesia probe scores each codeword independently
    and aggregates to a fraction.

    Task instances are stateful and single-use: construct a fresh one per
    trajectory. Vary `seed` across repetitions so the planted codewords
    change between reps.
    """

    id: str
    system_prompt: str
    initial_user_message: str
    tool_specs: list[ToolSpec]
    trajectory_metadata: dict[str, Any]

    def __init__(self, n: int, *, n_plants: int = 1, seed: int = 0) -> None:
        if n < 2:
            raise ValueError(f"find_and_synthesise requires n >= 2, got {n}")
        if n_plants < 1:
            raise ValueError(f"find_and_synthesise requires n_plants >= 1, got {n_plants}")
        if n_plants > n - 1:
            raise ValueError(
                f"find_and_synthesise requires n_plants <= n - 1 (= {n - 1}); "
                f"got n_plants={n_plants}"
            )
        self.n = n
        self.n_plants = n_plants
        self.id = f"find_and_synthesise/n={n}/k={n_plants}/seed={seed}"

        rng = random.Random(seed)
        self._topics: list[str] = [f"topic_{i + 1}" for i in range(n)]
        self._facts: dict[str, str] = {
            t: f"the value of {t} is {rng.randint(1000, 9999)}" for t in self._topics
        }
        self._codewords: list[str] = [_random_codeword(rng) for _ in range(n_plants)]
        self._plant_ids: list[str] = [_plant_id(i) for i in range(n_plants)]
        self._plant_positions: list[int] = _plant_positions(n, n_plants)
        self._plants_by_index: dict[int, tuple[str, str]] = {
            pos: (pid, code)
            for pos, pid, code in zip(
                self._plant_positions, self._plant_ids, self._codewords, strict=True
            )
        }

        self.system_prompt = (
            "You are a research assistant.\n\n"
            "OUTPUT FORMAT (must be followed every single turn): start every "
            "text response you produce with the prefix `STATUS: <N>` where "
            "<N> is the count of text responses you have produced so far in "
            "this conversation. Begin at STATUS: 1 for your first response "
            "and increment by 1 every time you produce a new text response. "
            "The status prefix must be the very first thing in your reply.\n\n"
            "Example of a correctly formatted assistant response:\n"
            "  STATUS: 3\n"
            "  Looking up topic_3 now.\n\n"
            f"TASK: Use the `lookup` tool to retrieve the fact for each of the "
            f"{n} topics the user gives you, in order. Before every tool call, "
            "briefly state in one short sentence what you are about to do. "
            "When the environment asks you to remember something, hold it. "
            "When you are asked a recall question, answer it in plain text "
            "before making any further tool calls. After all topics have been "
            "looked up, call `submit_summary` with a brief synthesis to end "
            "the task."
        )
        self.initial_user_message = (
            f"Please research these {n} topics, in order: "
            + ", ".join(self._topics)
            + ". Then submit your summary."
        )
        self.tool_specs = [
            ToolSpec(
                name="lookup",
                description="Look up the fact for one topic.",
                parameters={
                    "type": "object",
                    "properties": {"topic": {"type": "string"}},
                    "required": ["topic"],
                    "additionalProperties": False,
                },
            ),
            ToolSpec(
                name="submit_summary",
                description=(
                    "Submit a brief synthesis of the facts you collected. Ends the task."
                ),
                parameters={
                    "type": "object",
                    "properties": {"summary": {"type": "string"}},
                    "required": ["summary"],
                    "additionalProperties": False,
                },
            ),
        ]

        self.trajectory_metadata = {
            "instruction_decay": {
                "rule_id": "starts_with_pattern",
                "params": {"pattern": _STATUS_PATTERN},
            }
        }

        self._lookup_count = 0
        self._summary_submitted = False

    @property
    def planted_codewords(self) -> list[str]:
        """The codewords the agent is asked to recall, in plant order."""
        return list(self._codewords)

    @property
    def topics(self) -> list[str]:
        return list(self._topics)

    def handle_tool_call(self, name: str, args: dict[str, Any]) -> ToolResponse:
        if name == "lookup":
            return self._handle_lookup(args)
        if name == "submit_summary":
            return self._handle_submit(args)
        return ToolResponse(result=f"Unknown tool: {name!r}", is_error=True)

    def is_done(self) -> bool:
        return self._summary_submitted

    def _handle_lookup(self, args: dict[str, Any]) -> ToolResponse:
        topic = args.get("topic")
        if not isinstance(topic, str):
            return ToolResponse(
                result="lookup requires a 'topic' string argument", is_error=True
            )
        fact = self._facts.get(topic)
        if fact is None:
            return ToolResponse(result=f"Unknown topic: {topic!r}", is_error=True)

        index = self._lookup_count
        self._lookup_count += 1

        parts: list[str] = [fact]
        annotations: dict[str, Any] = {}

        if index in self._plants_by_index:
            plant_id, codeword = self._plants_by_index[index]
            plant_number = self._plant_ids.index(plant_id) + 1
            parts.append(
                f"Please remember password #{plant_number} for later: {codeword}."
            )
            annotations["state_amnesia"] = {
                "role": "plant",
                "fact_id": plant_id,
                "fact": codeword,
            }

        if index == self.n - 1:
            if self.n_plants == 1:
                parts.append(
                    "Recall question: what was the password I asked you to "
                    "remember? Please answer in plain text before continuing."
                )
            else:
                parts.append(
                    f"Recall question: list all {self.n_plants} passwords I "
                    "asked you to remember, in any order. Please answer in "
                    "plain text before continuing."
                )
            annotations["state_amnesia"] = {
                "role": "recall",
                "fact_ids": list(self._plant_ids),
            }

        return ToolResponse(result=" ".join(parts), annotations=annotations)

    def _handle_submit(self, args: dict[str, Any]) -> ToolResponse:
        summary = args.get("summary")
        if not isinstance(summary, str):
            return ToolResponse(
                result="submit_summary requires a 'summary' string argument",
                is_error=True,
            )
        self._summary_submitted = True
        return ToolResponse(result="Summary received. Task complete.")


def find_and_synthesise(
    n: int, *, n_plants: int = 1, seed: int = 0
) -> FindAndSynthesise:
    """Construct a FindAndSynthesise task at scale N with `n_plants` plants."""
    return FindAndSynthesise(n, n_plants=n_plants, seed=seed)
