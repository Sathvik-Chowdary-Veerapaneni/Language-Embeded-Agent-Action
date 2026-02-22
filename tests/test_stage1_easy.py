"""
LEAA Test Suite — Stage 1: EASY

Basic sanity checks — does everything work at the most fundamental level?
These tests should ALWAYS pass. If any fail, something is seriously broken.

Tests:
    - Environment creation and space shapes
    - Reset returns valid observations
    - Step mechanics and episode termination
    - Reward zone correctness (bullseye, inner, outer, miss)
    - Basic physics (launch velocity, gravity, arrow types)
    - Target creation and collision detection basics
"""

import numpy as np
import pytest

from physics_engine.ballistics import (
    STANDARD_ARROW,
    HEAVY_ARROW,
    LIGHT_ARROW,
    ARROW_TYPES,
    WindModel,
    simulate_trajectory,
    compute_launch_velocity,
)
from physics_engine.collision import (
    Target,
    check_hit,
    compute_reward,
)
from rl_training.envs.archery_env import ArcheryEnv


# ============================================================
# 1. Environment Basics
# ============================================================

class TestEnvironmentCreation:
    """Test that environments can be created with correct spaces."""

    def test_default_env_creates(self, default_env):
        """Default environment should create without errors."""
        assert default_env is not None

    def test_observation_space_shape(self, default_env):
        """Observation space should be (22,) floats."""
        assert default_env.observation_space.shape == (22,)
        assert default_env.observation_space.dtype == np.float32

    def test_action_space_shape(self, default_env):
        """Action space should be (3,) floats in [-1, 1]."""
        assert default_env.action_space.shape == (3,)
        assert default_env.action_space.low.min() == -1.0
        assert default_env.action_space.high.max() == 1.0

    def test_custom_config_applied(self):
        """Custom stage config should override defaults."""
        cfg = {"target_radius": 5.0, "target_distance_range": [1, 2]}
        env = ArcheryEnv(stage_config=cfg)
        assert env.config["target_radius"] == 5.0
        assert env.config["target_distance_range"] == [1, 2]
        env.close()

    def test_all_arrow_types_exist(self):
        """Three arrow types should be defined."""
        assert len(ARROW_TYPES) == 3
        names = {a.name for a in ARROW_TYPES}
        assert names == {"standard", "heavy", "light"}


class TestEnvironmentReset:
    """Test reset behavior."""

    def test_reset_returns_obs_and_info(self, default_env):
        """Reset should return (obs, info) tuple."""
        result = default_env.reset(seed=42)
        assert len(result) == 2
        obs, info = result
        assert obs.shape == (22,)
        assert obs.dtype == np.float32

    def test_reset_info_contains_keys(self, default_env):
        """Info dict should contain expected keys."""
        _, info = default_env.reset(seed=42)
        assert "target_distance" in info
        assert "wind_speed" in info
        assert "arrow_type" in info

    def test_reset_obs_no_nan(self, default_env):
        """Observation should never contain NaN after reset."""
        for seed in range(50):
            obs, _ = default_env.reset(seed=seed)
            assert not np.any(np.isnan(obs)), f"NaN in obs at seed={seed}"

    def test_reset_obs_no_inf(self, default_env):
        """Observation should never contain Inf after reset."""
        for seed in range(50):
            obs, _ = default_env.reset(seed=seed)
            assert not np.any(np.isinf(obs)), f"Inf in obs at seed={seed}"

    def test_reset_with_different_seeds(self, default_env):
        """Different seeds should produce different observations."""
        obs1, _ = default_env.reset(seed=1)
        obs2, _ = default_env.reset(seed=999)
        assert not np.allclose(obs1, obs2), "Different seeds gave same obs"


class TestEnvironmentStep:
    """Test step behavior."""

    def test_step_returns_five_values(self, default_env):
        """Step should return (obs, reward, terminated, truncated, info)."""
        default_env.reset(seed=42)
        action = default_env.action_space.sample()
        result = default_env.step(action)
        assert len(result) == 5

    def test_episode_terminates_after_one_step(self, default_env):
        """Episode should always end after a single shot."""
        default_env.reset(seed=42)
        _, _, terminated, truncated, _ = default_env.step(np.array([0, 0, 0], dtype=np.float32))
        assert terminated is True
        assert truncated is False

    def test_step_obs_shape(self, default_env):
        """Observation from step should have correct shape."""
        default_env.reset(seed=42)
        obs, _, _, _, _ = default_env.step(np.array([0, 0, 0], dtype=np.float32))
        assert obs.shape == (22,)
        assert obs.dtype == np.float32

    def test_step_info_keys(self, default_env):
        """Step info should contain all expected keys."""
        default_env.reset(seed=42)
        _, _, _, _, info = default_env.step(np.array([0, 0, 0], dtype=np.float32))
        expected_keys = {"hit", "reward", "distance_from_center", "closest_approach",
                         "target_distance", "wind_speed", "arrow_type", "draw_strength"}
        assert expected_keys.issubset(info.keys())

    def test_action_clipping(self, default_env):
        """Actions outside [-1, 1] should be clipped, not crash."""
        default_env.reset(seed=42)
        extreme_action = np.array([5.0, -10.0, 100.0], dtype=np.float32)
        obs, reward, terminated, _, _ = default_env.step(extreme_action)
        assert terminated is True
        assert not np.any(np.isnan(obs))

    def test_reward_is_finite(self, default_env):
        """Reward should always be a finite number."""
        for seed in range(20):
            default_env.reset(seed=seed)
            action = default_env.action_space.sample()
            _, reward, _, _, _ = default_env.step(action)
            assert np.isfinite(reward), f"Non-finite reward at seed={seed}: {reward}"


# ============================================================
# 2. Reward System
# ============================================================

class TestRewardZones:
    """Test reward calculation correctness."""

    def test_bullseye_reward(self):
        """Bullseye (center 10%) should give 100 points."""
        assert compute_reward(True, 0.05, 1.0) == 100.0
        assert compute_reward(True, 0.0, 1.0) == 100.0
        assert compute_reward(True, 0.09, 1.0) == 100.0

    def test_inner_ring_reward(self):
        """Inner ring (10-50%) should give 50 points."""
        assert compute_reward(True, 0.11, 1.0) == 50.0
        assert compute_reward(True, 0.30, 1.0) == 50.0
        assert compute_reward(True, 0.49, 1.0) == 50.0

    def test_outer_ring_reward(self):
        """Outer ring (50-100%) should give 25 points."""
        assert compute_reward(True, 0.51, 1.0) == 25.0
        assert compute_reward(True, 0.80, 1.0) == 25.0
        assert compute_reward(True, 0.99, 1.0) == 25.0

    def test_miss_reward_near(self):
        """Near miss should give small positive reward."""
        reward = compute_reward(False, None, 1.0, closest_approach=2.0)
        assert 2.0 < reward < 5.0, f"Near miss reward: {reward}"

    def test_miss_reward_far(self):
        """Far miss should give near-zero reward."""
        reward = compute_reward(False, None, 1.0, closest_approach=30.0)
        assert reward < 0.01, f"Far miss should be ~0, got {reward}"

    def test_miss_reward_decreases_with_distance(self):
        """Miss reward should decrease as distance increases."""
        r1 = compute_reward(False, None, 1.0, closest_approach=1.0)
        r2 = compute_reward(False, None, 1.0, closest_approach=5.0)
        r3 = compute_reward(False, None, 1.0, closest_approach=20.0)
        assert r1 > r2 > r3

    def test_reward_scales_with_target_radius(self):
        """Bullseye reward should trigger relative to target size."""
        # 5% of a 2.0m target = 0.1m from center → bullseye
        assert compute_reward(True, 0.1, 2.0) == 100.0
        # 5% of a 0.5m target = 0.025m → bullseye
        assert compute_reward(True, 0.025, 0.5) == 100.0


# ============================================================
# 3. Basic Physics
# ============================================================

class TestLaunchVelocity:
    """Test arrow launch velocity computation."""

    def test_zero_draw_gives_zero_velocity(self):
        """Zero draw strength should produce zero velocity."""
        vel = compute_launch_velocity(np.radians(45), 0.0, 0.0, STANDARD_ARROW)
        assert np.allclose(vel, 0.0)

    def test_max_draw_gives_max_speed(self):
        """Max draw on standard arrow should give ~70 m/s."""
        vel = compute_launch_velocity(0.0, 0.0, 1.0, STANDARD_ARROW)
        speed = np.linalg.norm(vel)
        assert 60.0 <= speed <= 75.0, f"Expected ~70 m/s, got {speed}"

    def test_velocity_direction_matches_angles(self):
        """Aiming straight ahead should give velocity along x-axis."""
        vel = compute_launch_velocity(0.0, 0.0, 1.0, STANDARD_ARROW)
        assert vel[0] > 0  # Forward
        assert abs(vel[1]) < 0.01  # No lateral
        assert abs(vel[2]) < 0.01  # No vertical

    def test_pitch_up_gives_positive_z(self):
        """Pitching up should give positive z-velocity."""
        vel = compute_launch_velocity(np.radians(45), 0.0, 1.0, STANDARD_ARROW)
        assert vel[2] > 0

    def test_heavy_arrow_slower_than_light(self):
        """Heavy arrows should launch slower than light arrows."""
        v_heavy = np.linalg.norm(compute_launch_velocity(0, 0, 1.0, HEAVY_ARROW))
        v_light = np.linalg.norm(compute_launch_velocity(0, 0, 1.0, LIGHT_ARROW))
        assert v_light > v_heavy


class TestTrajectoryBasics:
    """Test basic trajectory simulation."""

    def test_trajectory_returns_list(self):
        """Trajectory should return a list of (pos, vel) tuples."""
        vel = compute_launch_velocity(np.radians(30), 0.0, 1.0, STANDARD_ARROW)
        traj = simulate_trajectory(np.zeros(3), vel, STANDARD_ARROW, WindModel())
        assert len(traj) > 1
        assert len(traj[0]) == 2  # (position, velocity)

    def test_arrow_falls_to_ground(self):
        """Arrow should eventually hit the ground (z <= 0)."""
        vel = compute_launch_velocity(np.radians(45), 0.0, 1.0, STANDARD_ARROW)
        traj = simulate_trajectory(np.array([0, 0, 1.0]), vel, STANDARD_ARROW, WindModel())
        final_z = traj[-1][0][2]
        assert final_z <= 0.01, f"Arrow didn't reach ground, final z={final_z}"

    def test_trajectory_starts_at_origin(self):
        """First trajectory point should be the launch origin."""
        origin = np.array([5.0, 3.0, 1.0])
        vel = compute_launch_velocity(np.radians(30), 0.0, 0.5, STANDARD_ARROW)
        traj = simulate_trajectory(origin, vel, STANDARD_ARROW, WindModel())
        assert np.allclose(traj[0][0], origin)

    def test_gravity_pulls_arrow_down(self):
        """Arrow fired horizontally should lose altitude continuously."""
        vel = compute_launch_velocity(0.0, 0.0, 1.0, STANDARD_ARROW)
        traj = simulate_trajectory(np.array([0, 0, 10.0]), vel, STANDARD_ARROW, WindModel())
        heights = [p[2] for p, _ in traj]
        # After initial point, height should generally decrease
        assert heights[-1] < heights[0]


class TestCollisionBasics:
    """Test basic hit detection."""

    def test_direct_hit_detected(self):
        """Trajectory passing through target center should register hit."""
        target = Target("t", position=np.array([10.0, 0.0, 1.0]), radius=2.0)
        # Simulate straight shot
        vel = compute_launch_velocity(0.0, 0.0, 0.5, STANDARD_ARROW)
        traj = simulate_trajectory(np.array([0, 0, 1.0]), vel, STANDARD_ARROW, WindModel())
        hit, pos, dist = check_hit(traj, target)
        assert hit is True
        assert pos is not None
        assert dist is not None
        assert dist <= target.radius + 1e-6  # Floating-point tolerance

    def test_miss_detected(self):
        """Trajectory far from target should not register hit."""
        target = Target("t", position=np.array([0.0, 100.0, 50.0]), radius=0.5)
        vel = compute_launch_velocity(np.radians(10), 0.0, 0.5, STANDARD_ARROW)
        traj = simulate_trajectory(np.array([0, 0, 1.0]), vel, STANDARD_ARROW, WindModel())
        hit, pos, dist = check_hit(traj, target)
        assert hit is False
        assert pos is None
