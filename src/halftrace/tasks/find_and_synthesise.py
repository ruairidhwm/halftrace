"""find_and_synthesise: the canonical agentic task for the context-rot study.

Two modes:

- **Sequential (default).** The agent is given the full list of N topics
  and asked to look up each one via the `lookup` tool, then call
  `submit_summary` to end the task.
- **Discovery (`discovery=True`).** The agent is NOT given the topic
  list. It must call `discover_next(seen=[...])` to reveal the next
  topic, then `lookup(topic)` to fetch its value, repeating until
  `discover_next` returns `DONE`. Each `discover_next` call must pass
  `seen` containing exactly the topics revealed so far; the server
  validates and errors on drift. This is the adversarial variant that
  forces the agent to maintain its own state.

`n_plants` codewords are injected at evenly-spaced *call positions* in
the relevant stream (lookups in sequential mode, discovery calls in
discovery mode). All planted codewords are recalled together in one
final question; the `state_amnesia` probe scores each codeword
independently and aggregates to a fraction.
"""

from __future__ import annotations

import random
from typing import Any, cast

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
    """Evenly-spaced indices in [0, n-1) for the planted facts.

    Recall fires on index ``n-1``; plants must precede it. Guaranteed
    unique whenever ``n_plants <= n - 1``.
    """
    return [i * (n - 1) // n_plants for i in range(n_plants)]


def _random_codeword(rng: random.Random) -> str:
    return f"{rng.choice(_CODEWORDS)}{rng.randint(10, 99)}"


class FindAndSynthesise:
    """N lookups + a submit, with one or more plants spread across the trajectory.

    See module docstring for the two modes. Task instances are stateful
    and single-use: construct a fresh one per trajectory. Vary `seed`
    across repetitions so the planted codewords change between reps.
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
        n_plants: int = 1,
        discovery: bool = False,
        seed: int = 0,
    ) -> None:
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
        self.discovery = discovery
        mode_tag = "discovery" if discovery else "sequential"
        self.id = (
            f"find_and_synthesise/{mode_tag}/n={n}/k={n_plants}/seed={seed}"
        )

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

        if discovery:
            # Shuffle topic discovery order so the agent cannot infer it from
            # the names alone.
            self._discovery_order = list(self._topics)
            rng.shuffle(self._discovery_order)

        self.system_prompt = self._build_system_prompt()
        self.initial_user_message = self._build_initial_user_message()
        self.tool_specs = self._build_tool_specs()

        self.trajectory_metadata = {
            "instruction_decay": {
                "rule_id": "starts_with_pattern",
                "params": {"pattern": _STATUS_PATTERN},
            }
        }

        self._lookup_count = 0
        self._discovery_count = 0
        self._discovered_so_far: list[str] = []
        self._summary_submitted = False

    def _build_system_prompt(self) -> str:
        base = (
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
        )
        if self.discovery:
            return base + (
                "TASK: You must discover the topics in this dataset by "
                "calling `discover_next` and then look up each one with "
                "`lookup`. Each call to `discover_next` requires a "
                "`seen` argument: the list of topic names that have "
                "already been revealed to you. Pass `seen=[]` on your "
                "first call. After every successful `discover_next`, "
                "use `lookup(topic)` on the topic you just received. "
                "Repeat until `discover_next` returns `DONE`. Before "
                "every tool call, briefly state in one short sentence "
                "what you are about to do. When the environment asks "
                "you to remember something, hold it. When you are asked "
                "a recall question, answer it in plain text before "
                "making any further tool calls. After discovery is "
                "DONE, call `submit_summary` with a brief synthesis to "
                "end the task."
            )
        return base + (
            f"TASK: Use the `lookup` tool to retrieve the fact for each of the "
            f"{self.n} topics the user gives you, in order. Before every tool "
            "call, briefly state in one short sentence what you are about to "
            "do. When the environment asks you to remember something, hold "
            "it. When you are asked a recall question, answer it in plain "
            "text before making any further tool calls. After all topics "
            "have been looked up, call `submit_summary` with a brief "
            "synthesis to end the task."
        )

    def _build_initial_user_message(self) -> str:
        if self.discovery:
            return (
                "I have a set of records to investigate. Use `discover_next` "
                "to find each topic (start with `seen=[]`), then `lookup` to "
                "fetch its value. Repeat until `discover_next` returns DONE, "
                "then submit your summary."
            )
        return (
            f"Please research these {self.n} topics, in order: "
            + ", ".join(self._topics)
            + ". Then submit your summary."
        )

    def _build_tool_specs(self) -> list[ToolSpec]:
        specs: list[ToolSpec] = [
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
        ]
        if self.discovery:
            specs.append(
                ToolSpec(
                    name="discover_next",
                    description=(
                        "Reveal the next undiscovered topic in this dataset. "
                        "`seen` must list every topic already revealed to you. "
                        "Returns `DONE` once all topics have been revealed."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "seen": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                        "required": ["seen"],
                        "additionalProperties": False,
                    },
                )
            )
        specs.append(
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
            )
        )
        return specs

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
        if name == "discover_next" and self.discovery:
            return self._handle_discover_next(args)
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

        # In discovery mode, plants and recall fire on `discover_next` rather
        # than on `lookup` — the agent's bookkeeping pressure is on discovery.
        if self.discovery:
            return ToolResponse(result=fact)

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

    def _handle_discover_next(self, args: dict[str, Any]) -> ToolResponse:
        raw_seen = args.get("seen")
        if not isinstance(raw_seen, list):
            return ToolResponse(
                result="discover_next requires a 'seen' list argument",
                is_error=True,
            )
        seen_list = cast("list[Any]", raw_seen)
        if not all(isinstance(item, str) for item in seen_list):
            return ToolResponse(
                result="discover_next 'seen' must be a list of strings",
                is_error=True,
            )
        seen: list[str] = cast("list[str]", seen_list)

        expected = set(self._discovered_so_far)
        provided: set[str] = set(seen)
        if provided != expected:
            missing: set[str] = expected - provided
            extra: set[str] = provided - expected
            parts = ["discover_next 'seen' is inconsistent with revealed history:"]
            if missing:
                parts.append(f"missing topics: {sorted(missing)}")
            if extra:
                parts.append(f"unexpected topics: {sorted(extra)}")
            return ToolResponse(result=" ".join(parts), is_error=True)

        index = self._discovery_count
        self._discovery_count += 1

        parts_out: list[str] = []
        annotations: dict[str, Any] = {}

        if index >= self.n:
            # All topics revealed; this is the recall trigger.
            parts_out.append("DONE — all topics have been revealed.")
            if self.n_plants == 1:
                parts_out.append(
                    "Recall question: what was the password I asked you to "
                    "remember? Please answer in plain text before continuing."
                )
            else:
                parts_out.append(
                    f"Recall question: list all {self.n_plants} passwords I "
                    "asked you to remember, in any order. Please answer in "
                    "plain text before continuing."
                )
            annotations["state_amnesia"] = {
                "role": "recall",
                "fact_ids": list(self._plant_ids),
            }
            return ToolResponse(result=" ".join(parts_out), annotations=annotations)

        topic = self._discovery_order[index]
        self._discovered_so_far.append(topic)
        parts_out.append(f"Next topic: {topic}.")

        if index in self._plants_by_index:
            plant_id, codeword = self._plants_by_index[index]
            plant_number = self._plant_ids.index(plant_id) + 1
            parts_out.append(
                f"Please remember password #{plant_number} for later: {codeword}."
            )
            annotations["state_amnesia"] = {
                "role": "plant",
                "fact_id": plant_id,
                "fact": codeword,
            }

        return ToolResponse(result=" ".join(parts_out), annotations=annotations)

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
    n: int,
    *,
    n_plants: int = 1,
    discovery: bool = False,
    seed: int = 0,
) -> FindAndSynthesise:
    """Construct a FindAndSynthesise task at scale N with `n_plants` plants."""
    return FindAndSynthesise(n, n_plants=n_plants, discovery=discovery, seed=seed)
