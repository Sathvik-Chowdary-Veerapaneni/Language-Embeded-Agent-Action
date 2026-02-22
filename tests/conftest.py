"""
LEAA Test Suite — Shared Fixtures

Provides reusable pytest fixtures for all test stages.
"""

import sys
import os
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from physics_engine.ballistics import (
    ArrowType,
    WindModel,
    STANDARD_ARROW,
    HEAVY_ARROW,
    LIGHT_ARROW,
    ARROW_TYPES,
    simulate_trajectory,
    compute_launch_velocity,
)
from physics_engine.collision import (
    Target,
    check_hit,
    compute_reward,
    move_targets,
)
from rl_training.envs.archery_env import ArcheryEnv, DEFAULT_STAGE_CONFIG


# ---------- Environment Fixtures ----------
@pytest.fixture
def default_env():
    """Basic archery environment with default config."""
    env = ArcheryEnv()
    yield env
    env.close()


@pytest.fixture
def close_range_env():
    """Easy: close range, static target, no wind, large target."""
    cfg = {
        "target_distance_range": [3, 10],
        "target_moving": False,
        "wind_enabled": False,
        "agent_moving": False,
        "target_radius": 2.0,
    }
    env = ArcheryEnv(stage_config=cfg)
    yield env
    env.close()


@pytest.fixture
def medium_env():
    """Medium: mid range, moving target, light wind."""
    cfg = {
        "target_distance_range": [10, 30],
        "target_moving": True,
        "target_speed_range": [1, 3],
        "wind_enabled": True,
        "wind_speed_range": [0, 5],
        "agent_moving": False,
        "target_radius": 1.5,
    }
    env = ArcheryEnv(stage_config=cfg)
    yield env
    env.close()


@pytest.fixture
def hard_env():
    """Hard: long range, fast moving, strong wind, small target, moving agent."""
    cfg = {
        "target_distance_range": [15, 50],
        "target_moving": True,
        "target_speed_range": [3, 8],
        "wind_enabled": True,
        "wind_speed_range": [5, 12],
        "agent_moving": True,
        "agent_speed": 2.0,
        "target_radius": 0.75,
    }
    env = ArcheryEnv(stage_config=cfg)
    yield env
    env.close()


# ---------- Physics Fixtures ----------
@pytest.fixture
def standard_target():
    """Static target at 20m."""
    return Target(
        id="test_target",
        position=np.array([20.0, 0.0, 1.5]),
        radius=1.0,
    )


@pytest.fixture
def moving_target():
    """Moving target at 30m."""
    return Target(
        id="moving_target",
        position=np.array([30.0, 0.0, 2.0]),
        radius=1.0,
        velocity=np.array([0.0, 2.0, 0.0]),
    )


@pytest.fixture
def no_wind():
    """Zero wind."""
    return WindModel()


@pytest.fixture
def strong_wind():
    """Strong crosswind."""
    return WindModel(
        direction=np.array([0.0, 1.0, 0.0]),
        speed=10.0,
        gust_variance=2.0,
    )


@pytest.fixture
def seeded_rng():
    """Seeded numpy RNG for determinism."""
    return np.random.default_rng(seed=42)
