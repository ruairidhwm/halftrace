"""Probes consume Trajectories and produce diagnostic Scores."""

from halftrace.probes.base import Probe, Score
from halftrace.probes.instruction_decay import instruction_decay
from halftrace.probes.narration_substitution import narration_substitution
from halftrace.probes.state_amnesia import state_amnesia
from halftrace.probes.tool_repetition import tool_repetition

__all__ = [
    "Probe",
    "Score",
    "instruction_decay",
    "narration_substitution",
    "state_amnesia",
    "tool_repetition",
]
