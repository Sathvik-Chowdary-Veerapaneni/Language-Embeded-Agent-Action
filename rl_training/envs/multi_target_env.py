"""
LEAA RL Training — Multi-Target Archery Environment

5 targets with unique flag colors. Agent must hit a specific target
identified by a target selector one-hot vector.

Observation space (~73 floats):
    Agent state (6) + 5 targets × 11 each (55) + wind (4) +
    arrow type (3) + target selector (5) = 73

This env is used AFTER the base policy is trained — fine-tuning on target selection.
"""

import sys
import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

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
from rl_training.envs.scene_registry import SceneRegistry, FLAG_COLORS


NUM_TARGETS = 5
OBS_DIM = 6 + (NUM_TARGETS * 11) + 4 + 3 + NUM_TARGETS  # = 73


class MultiTargetArcheryEnv(gym.Env):
    """Multi-target archery environment with 5 flagged targets.

    Agent receives a target selector indicating which target to hit.
    Reward: +100 correct, -50 wrong target, -1 miss.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        stage_config: dict = None,
        render_mode: str = None,
    ):
        super().__init__()

        self.config = {
            "target_distance_range": [10, 40],
            "target_moving": False,
            "target_speed_range": [0, 0],
            "wind_enabled": False,
            "wind_speed_range": [0, 0],
            "agent_moving": False,
            "target_radius": 1.0,
            **(stage_config or {}),
        }
        self.render_mode = render_mode

        # Observation: 73 floats
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

        # Action: aim_pitch, aim_yaw, draw_strength
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # State
        self.agent_position: np.ndarray = np.zeros(3)
        self.agent_forward: np.ndarray = np.array([1.0, 0.0, 0.0])
        self.registry = SceneRegistry()
        self.targets: list = []
        self.wind = WindModel()
        self.arrow_type: ArrowType = STANDARD_ARROW
        self.arrow_type_index: int = 0
        self.current_target_id: str = "obj_0"
        self.last_trajectory = None

        # Stats
        self.episode_count: int = 0
        self.correct_hit_count: int = 0

    def _spawn_targets(self) -> None:
        """Spawn 5 targets with unique flag colors at random positions."""
        self.registry.clear()
        self.targets = []

        d_min, d_max = self.config["target_distance_range"]

        for i in range(NUM_TARGETS):
            distance = self.np_random.uniform(d_min, d_max)
            angle = self.np_random.uniform(-np.pi / 2, np.pi / 2)
            height = self.np_random.uniform(0.5, 4.0)

            pos = self.agent_position + np.array([
                distance * np.cos(angle),
                distance * np.sin(angle),
                height,
            ])

            vel = np.zeros(3)
            if self.config["target_moving"]:
                sp_min, sp_max = self.config.get("target_speed_range", [1, 3])
                speed = self.np_random.uniform(sp_min, sp_max)
                direction = self.np_random.normal(size=3)
                direction[2] *= 0.3
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                vel = direction * speed

            target = Target(
                id=f"obj_{i}",
                position=pos,
                radius=self.config.get("target_radius", 1.0),
                velocity=vel,
                flag_color=FLAG_COLORS[i],
            )
            self.targets.append(target)

            self.registry.add_object(
                id=target.id,
                position=pos,
                velocity=vel,
                flag_color=target.flag_color,
                radius=target.radius,
            )

    def _randomize_wind(self) -> None:
        """Set wind conditions."""
        if self.config["wind_enabled"]:
            sp_min, sp_max = self.config.get("wind_speed_range", [0, 5])
            speed = self.np_random.uniform(sp_min, sp_max)
            direction = self.np_random.normal(size=3)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            self.wind = WindModel(direction=direction, speed=speed, gust_variance=speed * 0.1)
        else:
            self.wind = WindModel()

    def _get_observation(self) -> np.ndarray:
        """Build the 73-element observation vector."""
        obs_parts = []

        # Agent state (6)
        obs_parts.append(self.agent_position)                  # 3
        obs_parts.append(self.agent_forward)                   # 3

        # Per-target: position(3) + velocity(3) + color_onehot(5) = 11 × 5 = 55
        for i in range(NUM_TARGETS):
            t = self.targets[i]
            obs_parts.append(t.position)                       # 3
            obs_parts.append(t.velocity)                       # 3
            color_onehot = np.zeros(5)
            if t.flag_color in FLAG_COLORS:
                color_onehot[FLAG_COLORS.index(t.flag_color)] = 1.0
            obs_parts.append(color_onehot)                     # 5

        # Wind (4)
        obs_parts.append(self.wind.direction)                  # 3
        obs_parts.append([self.wind.speed])                    # 1

        # Arrow type one-hot (3)
        arrow_onehot = np.zeros(3)
        arrow_onehot[self.arrow_type_index] = 1.0
        obs_parts.append(arrow_onehot)                         # 3

        # Target selector one-hot (5)
        target_selector = np.zeros(NUM_TARGETS)
        target_idx = int(self.current_target_id.split("_")[1])
        target_selector[target_idx] = 1.0
        obs_parts.append(target_selector)                      # 5

        obs = np.concatenate(obs_parts).astype(np.float32)     # Total: 73
        return obs

    def reset(self, seed=None, options=None):
        """Reset environment. options can contain 'target_id' to specify which target to hit."""
        super().reset(seed=seed)

        # Agent position
        if self.config.get("agent_moving"):
            self.agent_position = np.array([
                self.np_random.uniform(-5, 5),
                self.np_random.uniform(-5, 5),
                1.0,
            ])
        else:
            self.agent_position = np.array([0.0, 0.0, 1.0])
        self.agent_forward = np.array([1.0, 0.0, 0.0])

        # Spawn targets
        self._spawn_targets()
        self._randomize_wind()

        # Arrow type
        self.arrow_type_index = self.np_random.integers(0, len(ARROW_TYPES))
        self.arrow_type = ARROW_TYPES[self.arrow_type_index]

        # Target selection
        if options and "target_id" in options:
            self.current_target_id = options["target_id"]
        else:
            self.current_target_id = f"obj_{self.np_random.integers(0, NUM_TARGETS)}"

        obs = self._get_observation()
        info = {
            "current_target": self.current_target_id,
            "target_color": self.targets[int(self.current_target_id.split("_")[1])].flag_color,
            "num_targets": NUM_TARGETS,
        }

        return obs, info

    def step(self, action: np.ndarray):
        """Execute one shot at the scene with 5 targets."""
        action = np.clip(action, -1.0, 1.0)

        # Map actions
        pitch = action[0] * np.radians(90)
        yaw_offset = action[1] * np.radians(45)

        # Bearing to the SELECTED target
        target_idx = int(self.current_target_id.split("_")[1])
        selected_target = self.targets[target_idx]
        to_target = selected_target.position - self.agent_position
        target_bearing = np.arctan2(to_target[1], to_target[0])
        yaw = target_bearing + yaw_offset

        # Draw strength
        draw_strength = 0.3 + (action[2] + 1.0) * 0.35

        # Launch
        velocity = compute_launch_velocity(pitch, yaw, draw_strength, self.arrow_type)
        trajectory = simulate_trajectory(
            self.agent_position.copy(), velocity, self.arrow_type, self.wind
        )
        self.last_trajectory = trajectory

        # Check which target (if any) was hit
        hit_target_id = None
        hit_pos = None
        hit_dist = None

        for t in self.targets:
            h, hp, d = check_hit(trajectory, t)
            if h:
                hit_target_id = t.id
                hit_pos = hp
                hit_dist = d
                break  # First hit wins (trajectory order)

        # Compute reward
        if hit_target_id is None:
            reward = -1.0  # Miss
        elif hit_target_id == self.current_target_id:
            reward = compute_reward(True, hit_dist, selected_target.radius)
            self.correct_hit_count += 1
        else:
            reward = -50.0  # Hit wrong target

        self.episode_count += 1

        terminated = True
        truncated = False
        obs = self._get_observation()
        info = {
            "hit_target": hit_target_id,
            "correct_target": self.current_target_id,
            "correct_hit": hit_target_id == self.current_target_id if hit_target_id else False,
            "reward": reward,
            "distance_from_center": hit_dist,
        }

        return obs, reward, terminated, truncated, info

    @property
    def correct_hit_rate(self) -> float:
        if self.episode_count == 0:
            return 0.0
        return self.correct_hit_count / self.episode_count


# ---------- Smoke Test ----------
if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("\n[bold cyan]═══ LEAA Multi-Target Environment Smoke Test ═══[/bold cyan]\n")

    # Test 1: Create and check spaces
    console.print("[bold]Test 1:[/bold] Environment creation")
    env = MultiTargetArcheryEnv()
    console.print(f"  Observation space: {env.observation_space}")
    console.print(f"  Action space: {env.action_space}")
    assert env.observation_space.shape == (OBS_DIM,), f"Expected ({OBS_DIM},), got {env.observation_space.shape}"
    console.print(f"  ✅ Obs dim = {OBS_DIM}")

    # Test 2: Reset
    console.print("\n[bold]Test 2:[/bold] Reset with target selection")
    obs, info = env.reset(seed=42, options={"target_id": "obj_2"})
    console.print(f"  Obs shape: {obs.shape}")
    console.print(f"  Target: {info['current_target']} ({info['target_color']})")
    assert info["current_target"] == "obj_2"
    assert info["target_color"] == "yellow"
    console.print("  ✅ Target selection works")

    # Test 3: Step
    console.print("\n[bold]Test 3:[/bold] Step with random action")
    action = env.action_space.sample()
    obs2, reward, term, trunc, info2 = env.step(action)
    console.print(f"  Reward: {reward:.1f}")
    console.print(f"  Hit target: {info2['hit_target']}")
    console.print(f"  Correct hit: {info2['correct_hit']}")
    assert term is True
    console.print("  ✅ Step returns valid output")

    # Test 4: Run 50 episodes
    console.print("\n[bold]Test 4:[/bold] 50 random episodes")
    env2 = MultiTargetArcheryEnv(stage_config={"target_radius": 2.0})
    results = {"miss": 0, "correct": 0, "wrong": 0}
    for i in range(50):
        obs, info = env2.reset(seed=i)
        action = env2.action_space.sample()
        _, r, _, _, info = env2.step(action)
        if info["hit_target"] is None:
            results["miss"] += 1
        elif info["correct_hit"]:
            results["correct"] += 1
        else:
            results["wrong"] += 1

    table = Table(title="Random Agent Results (50 episodes)")
    table.add_column("Outcome", style="cyan")
    table.add_column("Count")
    table.add_row("Correct target", str(results["correct"]))
    table.add_row("Wrong target", str(results["wrong"]))
    table.add_row("Miss", str(results["miss"]))
    console.print(table)
    console.print("  ✅ 50 episodes completed without errors")

    # Test 5: All 5 targets visible in scene
    console.print("\n[bold]Test 5:[/bold] Scene registry integration")
    obs, info = env2.reset(seed=99)
    assert env2.registry.count == 5
    for color in FLAG_COLORS:
        objs = env2.registry.get_by_color(color)
        assert len(objs) == 1, f"Expected 1 {color} target, got {len(objs)}"
    console.print("  ✅ All 5 flag colors present in registry")

    console.print("\n[bold green]All multi-target environment tests passed![/bold green]\n")
