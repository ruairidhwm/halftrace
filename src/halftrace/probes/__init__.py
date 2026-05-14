"""Probes consume Trajectories and produce diagnostic Scores."""

from halftrace.probes.base import Probe, Score
from halftrace.probes.instruction_decay import instruction_decay
from halftrace.probes.state_amnesia import state_amnesia

__all__ = ["Probe", "Score", "instruction_decay", "state_amnesia"]
