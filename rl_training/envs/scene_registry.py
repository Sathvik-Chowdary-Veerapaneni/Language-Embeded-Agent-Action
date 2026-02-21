"""
LEAA RL Training — Scene Object Registry

Object registry that the language layer will later query.
Maintains the scene state: positions, velocities, colors, types of all objects.
Provides queries like get_by_color, get_closest_to, etc.
"""

import json
import sys
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

FLAG_COLORS = ["red", "blue", "yellow", "green", "white"]


@dataclass
class SceneObject:
    """An object in the scene."""
    id: str
    position: np.ndarray
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    flag_color: str = "red"
    radius: float = 0.5
    obj_type: str = "target"
    is_active: bool = True

    def to_dict(self) -> dict:
        """JSON-serializable representation."""
        return {
            "id": self.id,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "flag_color": self.flag_color,
            "radius": self.radius,
            "obj_type": self.obj_type,
            "is_active": self.is_active,
        }


class SceneRegistry:
    """Manages all objects in the scene.

    Provides query methods that the language grounding layer will later use
    to resolve natural language references like 'the red flag' or 'the closest target'.
    """

    def __init__(self):
        self._objects: Dict[str, SceneObject] = {}

    def add_object(
        self,
        id: str,
        position: np.ndarray,
        velocity: np.ndarray = None,
        flag_color: str = "red",
        radius: float = 0.5,
        obj_type: str = "target",
    ) -> SceneObject:
        """Add an object to the scene."""
        obj = SceneObject(
            id=id,
            position=np.array(position, dtype=np.float64),
            velocity=np.array(velocity, dtype=np.float64) if velocity is not None else np.zeros(3),
            flag_color=flag_color,
            radius=radius,
            obj_type=obj_type,
            is_active=True,
        )
        self._objects[id] = obj
        return obj

    def remove_object(self, id: str) -> bool:
        """Remove an object by ID. Returns True if found and removed."""
        if id in self._objects:
            del self._objects[id]
            return True
        return False

    def get_by_id(self, id: str) -> Optional[SceneObject]:
        """Get a specific object by ID."""
        return self._objects.get(id)

    def get_by_color(self, color: str) -> List[SceneObject]:
        """Get all active objects with a given flag color."""
        return [
            obj for obj in self._objects.values()
            if obj.flag_color == color and obj.is_active
        ]

    def get_closest_to(self, position: np.ndarray) -> Optional[SceneObject]:
        """Get the active object closest to a given position."""
        active = [o for o in self._objects.values() if o.is_active]
        if not active:
            return None
        position = np.array(position, dtype=np.float64)
        return min(active, key=lambda o: np.linalg.norm(o.position - position))

    def get_fastest(self) -> Optional[SceneObject]:
        """Get the active object with highest velocity magnitude."""
        active = [o for o in self._objects.values() if o.is_active]
        if not active:
            return None
        return max(active, key=lambda o: np.linalg.norm(o.velocity))

    def get_all_active(self) -> List[SceneObject]:
        """Get all active objects."""
        return [o for o in self._objects.values() if o.is_active]

    @property
    def count(self) -> int:
        """Number of objects in registry."""
        return len(self._objects)

    def to_dict(self) -> dict:
        """JSON-serializable scene state (fed to LLM later)."""
        return {
            "objects": [obj.to_dict() for obj in self._objects.values()],
            "count": self.count,
        }

    def to_json(self) -> str:
        """JSON string of scene state."""
        return json.dumps(self.to_dict(), indent=2)

    def to_observation(self, target_id: str) -> np.ndarray:
        """Convert scene state to numpy array matching multi_target_env observation format.

        Per-target block (11 floats each):
            position (3) + velocity (3) + flag_color_onehot (5)

        Args:
            target_id: Which object is the current target (for selector one-hot).

        Returns:
            Numpy array of scene observations.
        """
        active = sorted(self.get_all_active(), key=lambda o: o.id)

        obs_parts = []
        target_selector = np.zeros(len(active))

        for i, obj in enumerate(active):
            obs_parts.append(obj.position)
            obs_parts.append(obj.velocity)

            # Flag color one-hot (5 colors)
            color_onehot = np.zeros(5)
            if obj.flag_color in FLAG_COLORS:
                color_onehot[FLAG_COLORS.index(obj.flag_color)] = 1.0
            obs_parts.append(color_onehot)

            if obj.id == target_id:
                target_selector[i] = 1.0

        obs_parts.append(target_selector)

        return np.concatenate(obs_parts).astype(np.float32)

    def clear(self) -> None:
        """Remove all objects."""
        self._objects.clear()


# ---------- Smoke Test ----------
if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("\n[bold cyan]═══ LEAA Scene Registry Smoke Test ═══[/bold cyan]\n")

    registry = SceneRegistry()

    # Add 5 targets
    colors = FLAG_COLORS
    for i in range(5):
        pos = np.random.uniform(-50, 50, size=3)
        pos[2] = abs(pos[2])  # Keep z positive
        vel = np.random.uniform(-3, 3, size=3) if i > 2 else np.zeros(3)
        registry.add_object(
            id=f"obj_{i}",
            position=pos,
            velocity=vel,
            flag_color=colors[i],
            radius=0.5 + i * 0.1,
        )

    # Test 1: Count
    console.print("[bold]Test 1:[/bold] Object count")
    assert registry.count == 5
    console.print(f"  ✅ Registry has {registry.count} objects")

    # Test 2: Get by color
    console.print("\n[bold]Test 2:[/bold] Query by color")
    reds = registry.get_by_color("red")
    assert len(reds) == 1
    console.print(f"  ✅ Found {len(reds)} red object: {reds[0].id}")

    # Test 3: Get by ID
    console.print("\n[bold]Test 3:[/bold] Query by ID")
    obj = registry.get_by_id("obj_2")
    assert obj is not None
    assert obj.flag_color == "yellow"
    console.print(f"  ✅ obj_2 → color={obj.flag_color}")

    # Test 4: Closest to origin
    console.print("\n[bold]Test 4:[/bold] Get closest to origin")
    closest = registry.get_closest_to(np.zeros(3))
    console.print(f"  ✅ Closest to origin: {closest.id} at {closest.position}")

    # Test 5: Fastest
    console.print("\n[bold]Test 5:[/bold] Get fastest")
    fastest = registry.get_fastest()
    fastest_speed = np.linalg.norm(fastest.velocity)
    console.print(f"  ✅ Fastest: {fastest.id} at {fastest_speed:.1f} m/s")

    # Test 6: to_dict (JSON serializable)
    console.print("\n[bold]Test 6:[/bold] JSON serialization")
    scene_dict = registry.to_dict()
    json_str = registry.to_json()
    assert "objects" in scene_dict
    assert len(scene_dict["objects"]) == 5
    console.print(f"  ✅ JSON output: {len(json_str)} chars, {scene_dict['count']} objects")

    # Test 7: to_observation
    console.print("\n[bold]Test 7:[/bold] Observation vector")
    obs = registry.to_observation("obj_2")
    expected_len = 5 * (3 + 3 + 5) + 5  # 5 targets * 11 each + 5 selector
    assert obs.shape == (expected_len,), f"Expected ({expected_len},), got {obs.shape}"
    assert obs.dtype == np.float32
    console.print(f"  ✅ Observation shape: {obs.shape}, dtype: {obs.dtype}")

    # Test 8: Remove object
    console.print("\n[bold]Test 8:[/bold] Remove object")
    removed = registry.remove_object("obj_4")
    assert removed
    assert registry.count == 4
    assert registry.get_by_id("obj_4") is None
    console.print(f"  ✅ Removed obj_4, count now: {registry.count}")

    # Print scene table
    console.print("\n")
    table = Table(title="Scene State")
    table.add_column("ID", style="cyan")
    table.add_column("Color", style="magenta")
    table.add_column("Position")
    table.add_column("Speed (m/s)")
    for obj in registry.get_all_active():
        table.add_row(
            obj.id,
            obj.flag_color,
            f"({obj.position[0]:.1f}, {obj.position[1]:.1f}, {obj.position[2]:.1f})",
            f"{np.linalg.norm(obj.velocity):.1f}",
        )
    console.print(table)

    console.print("\n[bold green]All scene registry tests passed![/bold green]\n")
