"""
LEAA Test Suite — Stage 3: HARD

Edge cases, stress tests, and training robustness.
These tests probe boundary conditions, numerical stability, and
training infrastructure resilience.

Tests:
    - NaN/Inf observation handling
    - Extreme action values don't crash
    - Extreme environmental conditions
    - Zero and very large distances
    - Stage transition safety (the NaN bug)
    - VecNormalize save/load round-trip
    - Callback logic (stage advancement, checkpointing)
    - Memory stability over many episodes
    - Training quick-test mode
    - Physics edge cases (zero velocity, vertical shots)
"""

import os
import sys
import tempfile
from collections import deque
from pathlib import Path

import numpy as np
import pytest
import torch

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


# ============================================================
# 1. Numerical Edge Cases
# ============================================================

class TestNumericalEdgeCases:
    """Test handling of extreme numerical values."""

    def test_nan_in_obs_clamped(self):
        """NaN values in raw observation should be replaced with 0."""
        env = ArcheryEnv()
        env.reset(seed=42)
        # Manually corrupt target position to produce NaN
        env.target.position = np.array([np.nan, np.nan, np.nan])
        obs = env._get_observation()
        assert not np.any(np.isnan(obs)), "NaN leaked into observation"
        env.close()

    def test_inf_in_obs_clamped(self):
        """Inf values in raw observation should be clamped."""
        env = ArcheryEnv()
        env.reset(seed=42)
        env.target.position = np.array([np.inf, -np.inf, np.inf])
        obs = env._get_observation()
        assert not np.any(np.isinf(obs)), "Inf leaked into observation"
        assert np.all(np.abs(obs) <= 10.0), "Inf not clamped properly"
        env.close()

    def test_zero_distance_target(self):
        """Target at agent's position should not crash."""
        env = ArcheryEnv()
        env.reset(seed=42)
        env.target.position = env.agent_position.copy()
        obs = env._get_observation()
        assert not np.any(np.isnan(obs))
        # Step should also work
        _, reward, terminated, _, _ = env.step(np.array([0, 0, 0], dtype=np.float32))
        assert np.isfinite(reward)
        assert terminated is True
        env.close()

    def test_very_large_distance_target(self):
        """Target very far away should not crash."""
        cfg = {"target_distance_range": [1000, 1000], "target_radius": 0.5}
        env = ArcheryEnv(stage_config=cfg)
        obs, _ = env.reset(seed=42)
        assert not np.any(np.isnan(obs))
        _, reward, _, _, _ = env.step(np.array([0, 0, 1], dtype=np.float32))
        assert np.isfinite(reward)
        env.close()

    def test_very_small_target_radius(self):
        """Tiny target radius should not cause division by zero."""
        cfg = {"target_radius": 0.001, "target_distance_range": [5, 10]}
        env = ArcheryEnv(stage_config=cfg)
        obs, _ = env.reset(seed=42)
        _, reward, _, _, _ = env.step(np.array([0, 0, 0], dtype=np.float32))
        assert np.isfinite(reward)
        env.close()

    def test_zero_radius_target_reward(self):
        """Zero radius in reward should not divide by zero."""
        # compute_reward with zero radius
        r = compute_reward(True, 0.0, 0.001)
        assert np.isfinite(r)
        r2 = compute_reward(False, None, 0.001, closest_approach=5.0)
        assert np.isfinite(r2)


# ============================================================
# 2. Extreme Actions
# ============================================================

class TestExtremeActions:
    """Test that extreme or pathological actions don't crash."""

    def test_all_zeros_action(self, default_env):
        """All-zero action should work."""
        default_env.reset(seed=42)
        _, r, t, _, _ = default_env.step(np.array([0, 0, 0], dtype=np.float32))
        assert t is True
        assert np.isfinite(r)

    def test_all_ones_action(self, default_env):
        """All-ones action should work."""
        default_env.reset(seed=42)
        _, r, t, _, _ = default_env.step(np.array([1, 1, 1], dtype=np.float32))
        assert t is True
        assert np.isfinite(r)

    def test_all_negative_ones_action(self, default_env):
        """All-negative-ones action should work."""
        default_env.reset(seed=42)
        _, r, t, _, _ = default_env.step(np.array([-1, -1, -1], dtype=np.float32))
        assert t is True
        assert np.isfinite(r)

    def test_out_of_range_action_clipped(self, default_env):
        """Actions outside [-1,1] should be clipped, not crash."""
        default_env.reset(seed=42)
        _, r, t, _, _ = default_env.step(np.array([100, -100, 50], dtype=np.float32))
        assert t is True
        assert np.isfinite(r)

    def test_rapid_fire_100_random_actions(self, default_env):
        """100 random actions in rapid succession should all work."""
        for i in range(100):
            default_env.reset(seed=i)
            action = default_env.action_space.sample()
            obs, r, t, _, info = default_env.step(action)
            assert t is True
            assert np.isfinite(r)
            assert not np.any(np.isnan(obs))


# ============================================================
# 3. Extreme Environment Conditions
# ============================================================

class TestExtremeConditions:
    """Test under harsh environmental conditions."""

    def test_hurricane_wind(self):
        """Very strong wind should not crash trajectory simulation."""
        origin = np.array([0, 0, 1.0])
        vel = compute_launch_velocity(np.radians(30), 0.0, 1.0, STANDARD_ARROW)
        hurricane = WindModel(
            direction=np.array([1, 1, 1]) / np.sqrt(3),
            speed=50.0,
            gust_variance=10.0,
        )
        rng = np.random.default_rng(42)
        traj = simulate_trajectory(origin, vel, STANDARD_ARROW, hurricane, rng=rng)
        assert len(traj) > 1
        # All positions should be finite
        for pos, vel in traj:
            assert np.all(np.isfinite(pos)), f"Non-finite position in hurricane: {pos}"

    def test_many_targets_moving(self):
        """100 targets moving for 1000 steps should stay in bounds."""
        targets = [
            Target(f"t{i}",
                   position=np.random.uniform(-50, 50, size=3),
                   velocity=np.random.uniform(-5, 5, size=3))
            for i in range(100)
        ]
        for _ in range(1000):
            move_targets(targets, dt=0.1)
        for t in targets:
            assert np.all(np.abs(t.position) <= 100.0), \
                f"Target {t.id} out of bounds: {t.position}"

    def test_vertical_shot_straight_up(self):
        """Shooting straight up should not crash."""
        origin = np.array([0, 0, 1.0])
        vel = compute_launch_velocity(np.radians(89), 0.0, 1.0, STANDARD_ARROW)
        traj = simulate_trajectory(origin, vel, STANDARD_ARROW, WindModel())
        assert len(traj) > 10
        # Should go up and come back down
        max_height = max(p[2] for p, _ in traj)
        assert max_height > 10.0, "Vertical shot didn't go high"
        assert traj[-1][0][2] <= 0.01, "Arrow didn't return to ground"

    def test_horizontal_shot(self):
        """Shooting perfectly horizontal should not crash."""
        origin = np.array([0, 0, 1.0])
        vel = compute_launch_velocity(0.0, 0.0, 1.0, STANDARD_ARROW)
        traj = simulate_trajectory(origin, vel, STANDARD_ARROW, WindModel())
        assert len(traj) > 1
        assert traj[-1][0][0] > 0, "Arrow didn't move forward"


# ============================================================
# 4. Physics Edge Cases
# ============================================================

class TestPhysicsEdgeCases:
    """Test physics engine boundary conditions."""

    def test_zero_velocity_drops_down(self):
        """Arrow with zero velocity should just fall straight down."""
        origin = np.array([0, 0, 10.0])
        traj = simulate_trajectory(origin, np.zeros(3), STANDARD_ARROW, WindModel())
        final = traj[-1][0]
        assert final[2] <= 0.01, "Arrow didn't fall"
        assert abs(final[0]) < 1.0, "Arrow moved horizontally with zero velocity"

    def test_line_segment_sphere_exact_touch(self):
        """Arrow passing exactly at target radius edge should hit."""
        # Target at (10, 0, 1) with radius 1
        target = Target("t", position=np.array([10.0, 0.0, 1.0]), radius=1.0)
        # Trajectory passing at y=0.99 (just inside radius)
        trajectory = [
            (np.array([0.0, 0.99, 1.0]), np.array([10, 0, 0])),
            (np.array([20.0, 0.99, 1.0]), np.array([10, 0, 0])),
        ]
        hit, _, _ = check_hit(trajectory, target)
        assert hit is True, "Should hit when trajectory is within radius"

    def test_line_segment_sphere_clear_miss(self):
        """Arrow passing just outside target should miss."""
        target = Target("t", position=np.array([10.0, 0.0, 1.0]), radius=1.0)
        trajectory = [
            (np.array([0.0, 1.5, 1.0]), np.array([10, 0, 0])),
            (np.array([20.0, 1.5, 1.0]), np.array([10, 0, 0])),
        ]
        hit, _, _ = check_hit(trajectory, target)
        assert hit is False, "Should miss when trajectory is outside radius"

    def test_optimal_angle_with_drag(self):
        """Optimal angle with drag should be < 45° (unlike vacuum physics)."""
        origin = np.array([0, 0, 1.0])
        best_angle, best_range = 0, 0
        for deg in range(5, 80, 1):
            vel = compute_launch_velocity(np.radians(deg), 0.0, 1.0, STANDARD_ARROW)
            traj = simulate_trajectory(origin, vel, STANDARD_ARROW, WindModel())
            r = traj[-1][0][0]
            if r > best_range:
                best_angle, best_range = deg, r
        # With drag, optimal is typically 30-40°, not 45°
        assert 25 <= best_angle <= 45, f"Optimal angle {best_angle}° seems wrong"


# ============================================================
# 5. Callback Logic
# ============================================================

class TestCurriculumCallback:
    """Test the curriculum callback logic (without actual training)."""

    def test_callback_tracks_hits(self):
        """Callback should correctly track hit/miss history."""
        from rl_training.train import CurriculumCallback, load_curriculum
        stages = load_curriculum()
        cb = CurriculumCallback(
            stages=stages,
            current_stage=0,
            checkpoint_dir=Path(tempfile.mkdtemp()),
            num_envs=1,
        )
        # Simulate 10 hits, 10 misses
        for _ in range(10):
            cb.hit_history.append(1.0)
        for _ in range(10):
            cb.hit_history.append(0.0)
        assert cb.success_rate == 0.5

    def test_callback_empty_history(self):
        """Success rate with no episodes should be 0."""
        from rl_training.train import CurriculumCallback, load_curriculum
        stages = load_curriculum()
        cb = CurriculumCallback(
            stages=stages, current_stage=0,
            checkpoint_dir=Path(tempfile.mkdtemp()),
        )
        assert cb.success_rate == 0.0

    def test_callback_stage_advance_resets_tracking(self):
        """Advancing stage should clear hit history and reset counters."""
        from rl_training.train import CurriculumCallback, load_curriculum
        stages = load_curriculum()
        cb = CurriculumCallback(
            stages=stages, current_stage=0,
            checkpoint_dir=Path(tempfile.mkdtemp()),
        )
        cb.hit_history.extend([1.0] * 100)
        cb.stage_episode_count = 5000
        cb.best_success_rate = 0.95

        cb.advance_stage()
        assert cb.current_stage == 1
        assert len(cb.hit_history) == 0
        assert cb.best_success_rate == 0.0
        assert cb.stage_episode_count == 0
        assert cb.stage_complete is False

    def test_callback_window_size(self):
        """Hit history should respect window size limit."""
        from rl_training.train import CurriculumCallback, load_curriculum
        stages = load_curriculum()
        cb = CurriculumCallback(
            stages=stages, current_stage=0,
            checkpoint_dir=Path(tempfile.mkdtemp()),
            window_size=100,
        )
        # Add 200 items — should only keep last 100
        for _ in range(200):
            cb.hit_history.append(1.0)
        assert len(cb.hit_history) == 100


# ============================================================
# 6. VecNormalize Save/Load
# ============================================================

class TestVecNormalize:
    """Test VecNormalize statistics persistence."""

    def test_vecnormalize_save_load_roundtrip(self):
        """Saving and loading VecNormalize stats should preserve them."""
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from stable_baselines3.common.monitor import Monitor

        def make():
            env = ArcheryEnv()
            return Monitor(env)

        vec_env = DummyVecEnv([make])
        vn = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

        # Run a few steps to build stats
        obs = vn.reset()
        for _ in range(100):
            action = [vn.action_space.sample()]
            obs, rewards, dones, infos = vn.step(action)
            if dones[0]:
                obs = vn.reset()

        # Save
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            save_path = f.name
        vn.save(save_path)

        # Load into a fresh vec env
        vec_env2 = DummyVecEnv([make])
        vn2 = VecNormalize.load(save_path, vec_env2)

        # Compare stats
        assert np.allclose(vn.obs_rms.mean, vn2.obs_rms.mean, atol=1e-6)
        assert np.allclose(vn.obs_rms.var, vn2.obs_rms.var, atol=1e-6)

        vn.close()
        vn2.close()
        os.unlink(save_path)


# ============================================================
# 7. Memory & Stability Over Long Runs
# ============================================================

class TestLongRunStability:
    """Test stability over many episodes."""

    def test_1000_episodes_all_valid(self):
        """1000 episodes should all produce valid results."""
        env = ArcheryEnv(stage_config={
            "target_distance_range": [5, 30],
            "target_moving": True,
            "target_speed_range": [1, 5],
            "wind_enabled": True,
            "wind_speed_range": [0, 10],
            "target_radius": 1.0,
        })
        nan_count = 0
        inf_count = 0
        for i in range(1000):
            obs, _ = env.reset(seed=i)
            action = env.action_space.sample()
            obs2, reward, terminated, truncated, info = env.step(action)

            if np.any(np.isnan(obs2)):
                nan_count += 1
            if np.any(np.isinf(obs2)):
                inf_count += 1
            assert terminated is True
            assert np.isfinite(reward)

        assert nan_count == 0, f"{nan_count}/1000 episodes had NaN observations"
        assert inf_count == 0, f"{inf_count}/1000 episodes had Inf observations"
        env.close()

    def test_episode_count_increments(self):
        """Episode counter should correctly track all episodes."""
        env = ArcheryEnv()
        n = 200
        for i in range(n):
            env.reset(seed=i)
            env.step(env.action_space.sample())
        assert env.episode_count == n
        env.close()

    def test_reward_distribution_reasonable(self):
        """Reward distribution should be bounded and reasonable."""
        env = ArcheryEnv(stage_config={"target_distance_range": [5, 20], "target_radius": 1.5})
        rewards = []
        for i in range(500):
            env.reset(seed=i)
            _, r, _, _, _ = env.step(env.action_space.sample())
            rewards.append(r)

        rewards = np.array(rewards)
        assert rewards.min() >= 0, "Negative reward found"
        assert rewards.max() <= 100, "Reward exceeds max (100)"
        assert rewards.mean() > 0, "Mean reward should be positive"
        env.close()


# ============================================================
# 8. Training Infrastructure Quick-Test
# ============================================================

class TestTrainingInfrastructure:
    """Test training components work end-to-end (lightweight)."""

    def test_ppo_can_predict(self):
        """PPO model should be able to predict actions from observations."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor

        def make():
            return Monitor(ArcheryEnv())

        env = DummyVecEnv([make])
        model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=1, verbose=0)

        # Single prediction
        obs = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == (1, 3)

        # Stochastic prediction
        action2, _ = model.predict(obs, deterministic=False)
        assert action2.shape == (1, 3)

        env.close()

    def test_ppo_can_train_one_update(self):
        """PPO should complete at least one training update without error."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from stable_baselines3.common.monitor import Monitor

        def make():
            return Monitor(ArcheryEnv())

        env = DummyVecEnv([make])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

        model = PPO(
            "MlpPolicy", env,
            n_steps=64,
            batch_size=32,
            n_epochs=1,
            gamma=1.0,
            ent_coef=0.03,
            verbose=0,
        )

        # Train for 1 update (64 steps)
        model.learn(total_timesteps=64)

        # Verify model can still predict
        obs = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == (1, 3)
        assert np.all(np.isfinite(action))

        env.close()

    def test_model_save_load_roundtrip(self):
        """Model should survive save/load cycle."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor

        def make():
            return Monitor(ArcheryEnv())

        env = DummyVecEnv([make])
        model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, verbose=0)

        obs = env.reset()
        action_before, _ = model.predict(obs, deterministic=True)

        # Save and reload
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            path = f.name
        model.save(path)
        model2 = PPO.load(path, env=env)
        action_after, _ = model2.predict(obs, deterministic=True)

        assert np.allclose(action_before, action_after), \
            "Model predictions changed after save/load"

        env.close()
        os.unlink(path)
