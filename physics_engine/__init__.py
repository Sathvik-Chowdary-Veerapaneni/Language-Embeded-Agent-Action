"""
LEAA Physics Engine
Core arrow flight physics and collision detection.
"""

from physics_engine.ballistics import (
    ArrowType,
    WindModel,
    STANDARD_ARROW,
    HEAVY_ARROW,
    LIGHT_ARROW,
    simulate_trajectory,
    compute_launch_velocity,
)
from physics_engine.collision import (
    Target,
    check_hit,
    compute_reward,
    move_targets,
)

__all__ = [
    "ArrowType",
    "WindModel",
    "STANDARD_ARROW",
    "HEAVY_ARROW",
    "LIGHT_ARROW",
    "simulate_trajectory",
    "compute_launch_velocity",
    "Target",
    "check_hit",
    "compute_reward",
    "move_targets",
]
