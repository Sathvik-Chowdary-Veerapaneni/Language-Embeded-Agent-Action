"""
LEAA Test Suite — Stage 2: MEDIUM

Integration and behavior tests — do components work together correctly?
Tests interactions between physics, environment, and training infrastructure.

Tests:
    - Curriculum config loading and validation
    - Wind affects trajectory direction
    - Moving targets change position over time
    - Seeded RNG produces deterministic results
    - Target prediction for moving targets works
    - VecNormalize wrapping works correctly
    - Multiple episodes produce varied results
    - Arrow type affects flight characteristics
    - Draw strength maps correctly
    - Observation vector encodes state accurately
"""

import numpy as np
import pytest
import yaml
from pathlib import Path

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
    move_targets,
)
from rl_training.envs.archery_env import ArcheryEnv


CONFIGS_DIR = Path(__file__).resolve().parent.parent / "rl_training" / "configs"


# ============================================================
# 1. Curriculum & Configuration
# ============================================================

class TestCurriculumConfig:
    """Test curriculum configuration loading and validity."""

    def test_curriculum_file_exists(self):
        """Curriculum config should exist."""
        assert (CONFIGS_DIR / "curriculum.yaml").exists()

    def test_curriculum_has_stages(self):
        """Curriculum should contain a list of stages."""
        with open(CONFIGS_DIR / "curriculum.yaml") as f:
            data = yaml.safe_load(f)
        assert "stages" in data
        assert len(data["stages"]) >= 3

    def test_each_stage_has_required_keys(self):
        """Every stage must have name, distance range, and success threshold."""
        with open(CONFIGS_DIR / "curriculum.yaml") as f:
            stages = yaml.safe_load(f)["stages"]
        required = {"name", "target_distance_range", "target_radius", "success_threshold"}
        for stage in stages:
            missing = required - set(stage.keys())
            assert not missing, f"Stage '{stage.get('name', '?')}' missing: {missing}"

    def test_stages_increase_in_difficulty(self):
        """Later stages should have smaller targets or longer distances."""
        with open(CONFIGS_DIR / "curriculum.yaml") as f:
            stages = yaml.safe_load(f)["stages"]
        # First stage should have largest target
        assert stages[0]["target_radius"] >= stages[-1]["target_radius"]
        # Last stage should have longest max distance
        assert stages[-1]["target_distance_range"][1] >= stages[0]["target_distance_range"][1]

    def test_success_thresholds_are_valid(self):
        """Success thresholds should be between 0 and 1."""
        with open(CONFIGS_DIR / "curriculum.yaml") as f:
            stages = yaml.safe_load(f)["stages"]
        for stage in stages:
            t = stage["success_threshold"]
            assert 0.0 < t <= 1.0, f"Invalid threshold {t} in {stage['name']}"


# ============================================================
# 2. Wind Effects
# ============================================================

class TestWindEffects:
    """Test that wind actually impacts trajectory."""

    def test_crosswind_deflects_trajectory(self):
        """Strong crosswind should push arrow sideways (y-axis)."""
        origin = np.array([0, 0, 1.0])
        vel = compute_launch_velocity(np.radians(15), 0.0, 0.8, STANDARD_ARROW)

        # No wind
        traj_calm = simulate_trajectory(origin, vel, STANDARD_ARROW, WindModel())
        # Strong crosswind
        wind = WindModel(direction=np.array([0, 1, 0]), speed=15.0, gust_variance=0.0)
        traj_wind = simulate_trajectory(origin, vel, STANDARD_ARROW, wind)

        # Compare y-position at landing
        y_calm = traj_calm[-1][0][1]
        y_wind = traj_wind[-1][0][1]
        assert abs(y_wind) > abs(y_calm) + 0.5, \
            f"Wind didn't deflect: calm y={y_calm:.2f}, wind y={y_wind:.2f}"

    def test_headwind_reduces_range(self):
        """Headwind should reduce arrow range (x-distance)."""
        origin = np.array([0, 0, 1.0])
        vel = compute_launch_velocity(np.radians(30), 0.0, 1.0, STANDARD_ARROW)

        traj_calm = simulate_trajectory(origin, vel, STANDARD_ARROW, WindModel())
        headwind = WindModel(direction=np.array([-1, 0, 0]), speed=10.0, gust_variance=0.0)
        traj_head = simulate_trajectory(origin, vel, STANDARD_ARROW, headwind)

        range_calm = traj_calm[-1][0][0]
        range_head = traj_head[-1][0][0]
        assert range_head < range_calm, \
            f"Headwind didn't reduce range: calm={range_calm:.1f}, head={range_head:.1f}"

    def test_tailwind_increases_range(self):
        """Tailwind should increase arrow range."""
        origin = np.array([0, 0, 1.0])
        vel = compute_launch_velocity(np.radians(30), 0.0, 1.0, STANDARD_ARROW)

        traj_calm = simulate_trajectory(origin, vel, STANDARD_ARROW, WindModel())
        tailwind = WindModel(direction=np.array([1, 0, 0]), speed=10.0, gust_variance=0.0)
        traj_tail = simulate_trajectory(origin, vel, STANDARD_ARROW, tailwind)

        range_calm = traj_calm[-1][0][0]
        range_tail = traj_tail[-1][0][0]
        assert range_tail > range_calm

    def test_zero_wind_has_no_gust(self):
        """WindModel with speed=0 should produce zero vector."""
        wind = WindModel()
        vec = wind.get_wind_vector()
        assert np.allclose(vec, 0.0)


# ============================================================
# 3. Determinism & Seeded RNG
# ============================================================

class TestDeterminism:
    """Test that seeded RNG produces reproducible results."""

    def test_same_seed_same_obs(self, default_env):
        """Same seed should produce identical observations."""
        obs1, _ = default_env.reset(seed=42)
        obs2, _ = default_env.reset(seed=42)
        assert np.allclose(obs1, obs2)

    def test_same_seed_same_reward(self, default_env):
        """Same seed + same action should produce same reward."""
        default_env.reset(seed=42)
        action = np.array([0.3, -0.2, 0.5], dtype=np.float32)
        _, r1, _, _, _ = default_env.step(action)

        default_env.reset(seed=42)
        _, r2, _, _, _ = default_env.step(action)
        assert r1 == r2, f"Same seed gave different rewards: {r1} vs {r2}"

    def test_seeded_wind_is_deterministic(self):
        """Wind with seeded RNG should produce same gusts."""
        wind = WindModel(
            direction=np.array([1, 0, 0]),
            speed=5.0,
            gust_variance=2.0,
        )
        rng1 = np.random.default_rng(seed=123)
        rng2 = np.random.default_rng(seed=123)

        vec1 = wind.get_wind_vector(rng=rng1)
        vec2 = wind.get_wind_vector(rng=rng2)
        assert np.allclose(vec1, vec2), "Seeded wind not deterministic"

    def test_seeded_trajectory_deterministic(self):
        """Same seed should produce identical trajectory with gusty wind."""
        origin = np.array([0, 0, 1.0])
        vel = compute_launch_velocity(np.radians(20), 0.0, 0.7, STANDARD_ARROW)
        wind = WindModel(direction=np.array([1, 1, 0]), speed=5.0, gust_variance=3.0)

        rng1 = np.random.default_rng(seed=99)
        rng2 = np.random.default_rng(seed=99)

        traj1 = simulate_trajectory(origin, vel, STANDARD_ARROW, wind, rng=rng1)
        traj2 = simulate_trajectory(origin, vel, STANDARD_ARROW, wind, rng=rng2)

        # Compare landing positions
        assert np.allclose(traj1[-1][0], traj2[-1][0]), "Seeded trajectories differ"


# ============================================================
# 4. Moving Target Prediction
# ============================================================

class TestMovingTargets:
    """Test moving target mechanics."""

    def test_target_moves_with_velocity(self):
        """move_targets should update position based on velocity."""
        t = Target("t", position=np.array([10.0, 0.0, 1.0]),
                   velocity=np.array([5.0, 0.0, 0.0]))
        move_targets([t], dt=1.0)
        assert np.allclose(t.position, [15.0, 0.0, 1.0])

    def test_target_bounces_at_boundary(self):
        """Target should bounce when hitting boundary."""
        t = Target("t", position=np.array([99.0, 0.0, 1.0]),
                   velocity=np.array([10.0, 0.0, 0.0]))
        move_targets([t], dt=1.0, bounds=(100.0, 50.0, 30.0))
        # Should have bounced — velocity reversed
        assert t.velocity[0] < 0, "Target didn't bounce at boundary"

    def test_env_moving_target_has_velocity(self, medium_env):
        """Medium env should produce targets with non-zero velocity."""
        medium_env.reset(seed=42)
        vel_norm = np.linalg.norm(medium_env.target.velocity)
        assert vel_norm > 0, "Moving target has zero velocity"

    def test_env_static_target_has_zero_velocity(self, close_range_env):
        """Static env should produce targets with zero velocity."""
        close_range_env.reset(seed=42)
        vel_norm = np.linalg.norm(close_range_env.target.velocity)
        assert vel_norm == 0, "Static target has non-zero velocity"


# ============================================================
# 5. Observation Encoding
# ============================================================

class TestObservationEncoding:
    """Test that observations correctly encode the environment state."""

    def test_obs_contains_agent_position(self, default_env):
        """First 3 elements should be agent position."""
        default_env.reset(seed=42)
        obs = default_env._get_observation()
        assert np.allclose(obs[0:3], default_env.agent_position)

    def test_obs_contains_agent_forward(self, default_env):
        """Elements 3-5 should be agent forward direction."""
        default_env.reset(seed=42)
        obs = default_env._get_observation()
        assert np.allclose(obs[3:6], default_env.agent_forward)

    def test_obs_contains_target_position(self, default_env):
        """Elements 6-8 should be target position."""
        default_env.reset(seed=42)
        obs = default_env._get_observation()
        assert np.allclose(obs[6:9], default_env.target.position)

    def test_obs_arrow_onehot_sums_to_one(self, default_env):
        """Arrow type one-hot (elements 15-17) should sum to 1."""
        default_env.reset(seed=42)
        obs = default_env._get_observation()
        arrow_onehot = obs[15:18]
        assert np.isclose(arrow_onehot.sum(), 1.0)
        assert set(arrow_onehot.tolist()).issubset({0.0, 1.0})

    def test_obs_distance_is_positive(self, default_env):
        """Distance to target (element 20) should be positive."""
        default_env.reset(seed=42)
        obs = default_env._get_observation()
        assert obs[20] > 0, "Distance should be positive"


# ============================================================
# 6. Arrow Flight Characteristics
# ============================================================

class TestArrowCharacteristics:
    """Test that different arrows have distinct flight profiles."""

    def test_heavy_arrow_shorter_range(self):
        """Heavy arrow should have shorter range than standard."""
        origin = np.array([0, 0, 1.0])
        for arrow in [STANDARD_ARROW, HEAVY_ARROW]:
            vel = compute_launch_velocity(np.radians(30), 0.0, 1.0, arrow)
            traj = simulate_trajectory(origin, vel, arrow, WindModel())

        v_std = compute_launch_velocity(np.radians(30), 0.0, 1.0, STANDARD_ARROW)
        v_hvy = compute_launch_velocity(np.radians(30), 0.0, 1.0, HEAVY_ARROW)
        t_std = simulate_trajectory(origin, v_std, STANDARD_ARROW, WindModel())
        t_hvy = simulate_trajectory(origin, v_hvy, HEAVY_ARROW, WindModel())

        range_std = t_std[-1][0][0]
        range_hvy = t_hvy[-1][0][0]
        assert range_std > range_hvy

    def test_light_arrow_more_wind_affected(self):
        """Light arrow should be more affected by wind than heavy."""
        origin = np.array([0, 0, 1.0])
        wind = WindModel(direction=np.array([0, 1, 0]), speed=10.0, gust_variance=0.0)

        v_light = compute_launch_velocity(np.radians(20), 0.0, 1.0, LIGHT_ARROW)
        v_heavy = compute_launch_velocity(np.radians(20), 0.0, 1.0, HEAVY_ARROW)

        t_light = simulate_trajectory(origin, v_light, LIGHT_ARROW, wind)
        t_heavy = simulate_trajectory(origin, v_heavy, HEAVY_ARROW, wind)

        # Light arrow should deflect more in y
        y_light = abs(t_light[-1][0][1])
        y_heavy = abs(t_heavy[-1][0][1])
        assert y_light > y_heavy, \
            f"Light arrow not more wind-affected: light y={y_light:.1f}, heavy y={y_heavy:.1f}"

    def test_draw_strength_maps_correctly(self, default_env):
        """Action draw_strength [-1,1] should map to [0.3, 1.0]."""
        default_env.reset(seed=42)

        # Min draw: action[2] = -1 → 0.3
        default_env.step(np.array([0, 0, -1.0], dtype=np.float32))
        assert abs(default_env.draw_strength - 0.3) < 0.01

        default_env.reset(seed=42)
        # Max draw: action[2] = 1 → 1.0
        default_env.step(np.array([0, 0, 1.0], dtype=np.float32))
        assert abs(default_env.draw_strength - 1.0) < 0.01


# ============================================================
# 7. Multi-Episode Sessions
# ============================================================

class TestMultiEpisode:
    """Test running many episodes produces varied, valid results."""

    def test_100_episodes_no_errors(self, default_env):
        """100 episodes should complete without any errors."""
        for i in range(100):
            obs, _ = default_env.reset(seed=i)
            action = default_env.action_space.sample()
            obs2, reward, terminated, truncated, info = default_env.step(action)
            assert terminated is True
            assert np.isfinite(reward)
            assert not np.any(np.isnan(obs2))

    def test_rewards_have_variance(self, default_env):
        """Rewards across episodes should not all be the same."""
        rewards = []
        for i in range(50):
            default_env.reset(seed=i)
            action = default_env.action_space.sample()
            _, r, _, _, _ = default_env.step(action)
            rewards.append(r)
        assert np.std(rewards) > 0.1, "Rewards have no variance — all same"

    def test_success_rate_tracks_correctly(self, close_range_env):
        """ArcheryEnv.success_rate should match manual count."""
        hits = 0
        n = 50
        for i in range(n):
            close_range_env.reset(seed=i)
            action = close_range_env.action_space.sample()
            _, _, _, _, info = close_range_env.step(action)
            if info["hit"]:
                hits += 1
        expected = hits / n
        assert abs(close_range_env.success_rate - expected) < 0.01
