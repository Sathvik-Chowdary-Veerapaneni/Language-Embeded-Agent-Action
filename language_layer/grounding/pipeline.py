"""
LEAA Language Layer — Grounding Pipeline

End-to-end orchestration:
  NL command string  -->  ParsedCommand  -->  ResolvedTarget
"""

import sys
import os
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from language_layer.grounding.command_parser import CommandParser, ParsedCommand
from language_layer.grounding.target_resolver import TargetResolver, ResolvedTarget
from rl_training.envs.scene_registry import SceneRegistry


class GroundingPipeline:
    """Full language grounding pipeline: command string -> resolved target."""

    def __init__(self, use_llm_fallback: bool = False):
        self.parser = CommandParser(use_llm_fallback=use_llm_fallback)
        self.resolver = TargetResolver()

    def ground(
        self,
        command: str,
        registry: SceneRegistry,
        agent_pos: np.ndarray,
    ) -> Optional[ResolvedTarget]:
        """Parse a natural-language command and resolve it to a scene target.

        Args:
            command: Raw NL command string (e.g. "shoot the red target").
            registry: Current scene state with all objects.
            agent_pos: Agent world position [x, y, z].

        Returns:
            ResolvedTarget with target object, arrow type, and confidence,
            or None if resolution fails entirely.
        """
        parsed: ParsedCommand = self.parser.parse(command)
        resolved: Optional[ResolvedTarget] = self.resolver.resolve(
            parsed, registry, agent_pos,
        )
        return resolved


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def smoke_test() -> None:
    """Quick integration test with ~8 commands."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("\n[bold cyan]=== LEAA Grounding Pipeline Smoke Test ===[/bold cyan]\n")

    # Build a test scene
    registry = SceneRegistry()
    registry.add_object("obj_0", position=[10, 0, 1], velocity=[0, 0, 0], flag_color="red")
    registry.add_object("obj_1", position=[20, 5, 1], velocity=[3, 0, 0], flag_color="blue")
    registry.add_object("obj_2", position=[30, -5, 1], velocity=[0, 0, 0], flag_color="red")
    registry.add_object("obj_3", position=[5, 10, 1], velocity=[0, 6, 0], flag_color="yellow")
    registry.add_object("obj_4", position=[50, 0, 1], velocity=[1, 1, 0], flag_color="green")

    agent_pos = np.array([0.0, 0.0, 1.0])
    pipeline = GroundingPipeline(use_llm_fallback=False)

    test_commands = [
        ("shoot obj_2", "obj_2", "explicit_id"),
        ("hit the red target", "obj_0", "color"),           # closest red
        ("fire at the blue one", "obj_1", "color"),
        ("shoot the closest target", "obj_0", "spatial"),
        ("aim at the farthest target", "obj_4", "spatial"),
        ("hit the fastest one", "obj_3", "speed"),
        ("use ice arrow on the green flag", "obj_4", "color"),
        ("shoot the second red target", "obj_2", "color"),  # ordinal
    ]

    table = Table(title="Smoke Test Results")
    table.add_column("Command", style="cyan", width=40)
    table.add_column("Expected", style="green")
    table.add_column("Got", style="yellow")
    table.add_column("Method", style="magenta")
    table.add_column("Pass?")

    all_pass = True
    for command, expected_id, expected_method in test_commands:
        result = pipeline.ground(command, registry, agent_pos)
        got_id = result.target_id if result else "None"
        got_method = result.resolution_method if result else "None"
        passed = got_id == expected_id
        if not passed:
            all_pass = False
        table.add_row(
            command,
            expected_id,
            got_id,
            got_method,
            "[green]YES[/green]" if passed else "[red]NO[/red]",
        )

    console.print(table)

    if all_pass:
        console.print("\n[bold green]All smoke tests passed![/bold green]\n")
    else:
        console.print("\n[bold red]Some smoke tests failed.[/bold red]\n")


if __name__ == "__main__":
    smoke_test()
