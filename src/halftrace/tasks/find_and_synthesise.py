"""find_and_synthesise: the canonical agentic task for the context-rot study.

The agent is given N topics and asked to look up each one via the `lookup`
tool, then call `submit_summary` to end the task. The first lookup response
plants a fact (a randomised password) and the last lookup response asks the
agent to recall it. The agent's text response after the recall is what the
`state_amnesia` probe scores.

The task is parameterised by N, the number of lookup calls required.
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

_PLANT_ID = "password"


def _random_codeword(rng: random.Random) -> str:
    return f"{rng.choice(_CODEWORDS)}{rng.randint(10, 99)}"


class FindAndSynthesise:
    """N lookups + a submit, with one plant/recall pair across the trajectory.

    Construct directly (`FindAndSynthesise(n=10)`) or via the convenience
    factory `find_and_synthesise(10)`. `seed` controls the randomised topic
    facts and the planted codeword for reproducibility.

    Task instances are stateful and single-use: construct a fresh one per
    trajectory. Vary `seed` across repetitions so the planted codeword
    changes between reps.
    """

    id: str
    system_prompt: str
    initial_user_message: str
    tool_specs: list[ToolSpec]

    def __init__(self, n: int, *, seed: int = 0) -> None:
        if n < 2:
            raise ValueError(f"find_and_synthesise requires n >= 2, got {n}")
        self.n = n
        self.id = f"find_and_synthesise/n={n}/seed={seed}"

        rng = random.Random(seed)
        self._topics: list[str] = [f"topic_{i + 1}" for i in range(n)]
        self._facts: dict[str, str] = {
            t: f"the value of {t} is {rng.randint(1000, 9999)}" for t in self._topics
        }
        self._planted_codeword: str = _random_codeword(rng)

        self.system_prompt = (
            "You are a research assistant. Use the `lookup` tool to retrieve the "
            f"fact for each of the {n} topics the user gives you, in order. "
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

        self._lookup_count = 0
        self._summary_submitted = False

    @property
    def planted_codeword(self) -> str:
        """The codeword the agent is asked to recall."""
        return self._planted_codeword

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

        if index == 0:
            parts.append(
                f"Please remember this password for later: {self._planted_codeword}."
            )
            annotations["state_amnesia"] = {
                "role": "plant",
                "fact_id": _PLANT_ID,
                "fact": self._planted_codeword,
            }
        if index == self.n - 1:
            parts.append(
                "Recall question: what was the password I asked you to remember? "
                "Please answer in plain text before continuing."
            )
            annotations["state_amnesia"] = {"role": "recall", "fact_id": _PLANT_ID}

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


def find_and_synthesise(n: int, *, seed: int = 0) -> FindAndSynthesise:
    """Construct a FindAndSynthesise task at scale N."""
    return FindAndSynthesise(n, seed=seed)
