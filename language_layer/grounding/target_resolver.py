"""
LEAA Language Layer — Target Resolver

Resolves a ParsedCommand against a SceneRegistry to produce a concrete
ResolvedTarget that the RL policy can act on.

Resolution priority:
  1. Explicit ID  (obj_N)
  2. Color        (+ optional ordinal / spatial disambiguation)
  3. Spatial ref  (closest / farthest / leftmost / rightmost)
  4. Speed ref    (fastest / slowest / stationary)
  5. Fallback     → closest active object to agent
"""

from dataclasses import dataclass
from typing import Optional, List

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rl_training.envs.scene_registry import SceneObject, SceneRegistry
from language_layer.grounding.command_parser import ParsedCommand


@dataclass
class ResolvedTarget:
    """Result of resolving a parsed command against the scene."""

    target_id: str
    target_obj: SceneObject
    arrow_type: Optional[str] = None
    confidence: float = 0.0
    resolution_method: str = "unknown"


class TargetResolver:
    """Resolves a ParsedCommand to a specific SceneObject."""

    def resolve(
        self,
        parsed: ParsedCommand,
        registry: SceneRegistry,
        agent_pos: np.ndarray,
    ) -> Optional[ResolvedTarget]:
        """Resolve *parsed* against *registry*.

        Args:
            parsed: Structured command from CommandParser.
            registry: Current scene state.
            agent_pos: Agent world position (used for spatial queries).

        Returns:
            ResolvedTarget or None if no valid target found.
        """
        arrow = parsed.arrow_type

        # --- 1. Explicit ID ---
        if parsed.target_id:
            obj = registry.get_by_id(parsed.target_id)
            if obj and obj.is_active:
                return ResolvedTarget(
                    target_id=obj.id,
                    target_obj=obj,
                    arrow_type=arrow,
                    confidence=0.95,
                    resolution_method="explicit_id",
                )

        # --- 2. Color ---
        if parsed.target_color:
            candidates = registry.get_by_color(parsed.target_color)
            candidates = [c for c in candidates if c.is_active]
            if candidates:
                chosen = self._disambiguate(
                    candidates, parsed, agent_pos,
                )
                return ResolvedTarget(
                    target_id=chosen.id,
                    target_obj=chosen,
                    arrow_type=arrow,
                    confidence=0.85 if len(candidates) == 1 else 0.75,
                    resolution_method="color",
                )

        # --- 3. Spatial ref ---
        if parsed.spatial_ref:
            obj = self._resolve_spatial(parsed.spatial_ref, registry, agent_pos)
            if obj:
                return ResolvedTarget(
                    target_id=obj.id,
                    target_obj=obj,
                    arrow_type=arrow,
                    confidence=0.80,
                    resolution_method="spatial",
                )

        # --- 4. Speed ref ---
        if parsed.speed_ref:
            obj = self._resolve_speed(parsed.speed_ref, registry)
            if obj:
                return ResolvedTarget(
                    target_id=obj.id,
                    target_obj=obj,
                    arrow_type=arrow,
                    confidence=0.80,
                    resolution_method="speed",
                )

        # --- 5. Fallback: closest to agent ---
        fallback = registry.get_closest_to(agent_pos)
        if fallback:
            return ResolvedTarget(
                target_id=fallback.id,
                target_obj=fallback,
                arrow_type=arrow,
                confidence=0.40,
                resolution_method="fallback_closest",
            )

        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _disambiguate(
        self,
        candidates: List[SceneObject],
        parsed: ParsedCommand,
        agent_pos: np.ndarray,
    ) -> SceneObject:
        """Pick one object from multiple same-color candidates.

        Uses ordinal, spatial ref, speed ref, or defaults to closest.
        """
        if len(candidates) == 1:
            return candidates[0]

        # Ordinal: sort by distance and pick nth
        if parsed.ordinal is not None:
            sorted_by_dist = sorted(
                candidates,
                key=lambda o: np.linalg.norm(o.position - agent_pos),
            )
            idx = min(parsed.ordinal - 1, len(sorted_by_dist) - 1)
            return sorted_by_dist[max(idx, 0)]

        # Spatial disambiguation
        if parsed.spatial_ref:
            obj = self._resolve_spatial_from_list(
                parsed.spatial_ref, candidates, agent_pos,
            )
            if obj:
                return obj

        # Speed disambiguation
        if parsed.speed_ref:
            obj = self._resolve_speed_from_list(parsed.speed_ref, candidates)
            if obj:
                return obj

        # Default: closest to agent
        return min(
            candidates,
            key=lambda o: np.linalg.norm(o.position - agent_pos),
        )

    def _resolve_spatial(
        self,
        ref: str,
        registry: SceneRegistry,
        agent_pos: np.ndarray,
    ) -> Optional[SceneObject]:
        active = registry.get_all_active()
        if not active:
            return None
        return self._resolve_spatial_from_list(ref, active, agent_pos)

    def _resolve_spatial_from_list(
        self,
        ref: str,
        objects: List[SceneObject],
        agent_pos: np.ndarray,
    ) -> Optional[SceneObject]:
        if not objects:
            return None

        if ref == "closest":
            return min(objects, key=lambda o: np.linalg.norm(o.position - agent_pos))
        elif ref == "farthest":
            return max(objects, key=lambda o: np.linalg.norm(o.position - agent_pos))
        elif ref == "leftmost":
            # y-axis positive = left in LEAA coordinate system
            return max(objects, key=lambda o: o.position[1])
        elif ref == "rightmost":
            return min(objects, key=lambda o: o.position[1])
        return None

    def _resolve_speed(
        self, ref: str, registry: SceneRegistry,
    ) -> Optional[SceneObject]:
        active = registry.get_all_active()
        if not active:
            return None
        return self._resolve_speed_from_list(ref, active)

    def _resolve_speed_from_list(
        self, ref: str, objects: List[SceneObject],
    ) -> Optional[SceneObject]:
        if not objects:
            return None

        if ref == "fastest":
            return max(objects, key=lambda o: np.linalg.norm(o.velocity))
        elif ref == "slowest":
            return min(objects, key=lambda o: np.linalg.norm(o.velocity))
        elif ref == "stationary":
            still = [o for o in objects if np.linalg.norm(o.velocity) < 0.1]
            if still:
                return still[0]
            # Fall back to slowest if nothing truly stationary
            return min(objects, key=lambda o: np.linalg.norm(o.velocity))
        return None
