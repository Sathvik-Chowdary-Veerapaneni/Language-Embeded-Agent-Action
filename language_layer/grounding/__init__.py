"""
LEAA — Language Grounding Sub-package

Exports the three main components:
  - CommandParser:     NL command  ->  ParsedCommand
  - TargetResolver:    ParsedCommand + SceneRegistry  ->  ResolvedTarget
  - GroundingPipeline: End-to-end orchestration
"""

from language_layer.grounding.command_parser import CommandParser, ParsedCommand
from language_layer.grounding.target_resolver import TargetResolver, ResolvedTarget
from language_layer.grounding.pipeline import GroundingPipeline

__all__ = [
    "CommandParser",
    "ParsedCommand",
    "TargetResolver",
    "ResolvedTarget",
    "GroundingPipeline",
]
