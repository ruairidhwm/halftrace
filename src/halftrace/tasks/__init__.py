"""Tasks specify what an agent must accomplish and respond to its tool calls."""

from halftrace.tasks.base import Task, ToolResponse, ToolSpec
from halftrace.tasks.find_and_synthesise import FindAndSynthesise, find_and_synthesise
from halftrace.tasks.find_max import FindMax, find_max

__all__ = [
    "FindAndSynthesise",
    "FindMax",
    "Task",
    "ToolResponse",
    "ToolSpec",
    "find_and_synthesise",
    "find_max",
]
