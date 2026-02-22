"""
LEAA RL Training — Archery Gymnasium Environment

Single-target archery environment wrapping the physics engine.
Agent takes one shot per episode → gets reward → reset.

Observation space (22 floats):
    Agent position (3) + forward dir (3) + target pos (3) + target vel (3) +
    wind dir (3) + wind speed (1) + arrow type one-hot (3) +
    draw strength (1) + distance to target (1) + elevation diff (1)

Action space (3 floats):
    aim_pitch [-1,1], aim_yaw [-1,1], draw_strength [-1,1]
"""

import sys
import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Add project root to path for imports
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
)


# Default stage config (static, close range)
DEFAULT_STAGE_CONFIG = {
    "target_distance_range": [3, 10],
    "target_moving": False,
    "target_speed_range": [0, 0],
    "wind_enabled": False,
    "wind_speed_range": [0, 0],
    "agent_moving": False,
    "agent_speed": 0.0,
    "target_radius": 2.0,
}

# Target radius by curriculum stage type
TARGET_RADIUS_MAP = {
    "static_close": 0.5,
    "static_far": 0.75,
    "moving_slow": 1.0,
    "wind": 1.0,
    "full_dynamic": 1.0,
}


class ArcheryEnv(gym.Env):
    """Single-target archery environment.

    The agent takes one action (aim + draw), fires an arrow, and the episode
    ends immediately with the resulting reward.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        stage_config: dict = None,
        render_mode: str = None,
    ):
        super().__init__()

        self.config = {**DEFAULT_STAGE_CONFIG, **(stage_config or {})}
        self.render_mode = render_mode

        # Observation space: 22 floats
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32
        )

        # Action space: 3 continuous values in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # State
        self.agent_position: np.ndarray = np.zeros(3)
        self.agent_forward: np.ndarray = np.array([1.0, 0.0, 0.0])
        self.target: Target = None
        self.wind: WindModel = WindModel()
        self.arrow_type: ArrowType = STANDARD_ARROW
        self.arrow_type_index: int = 0
        self.draw_strength: float = 0.5

        # Stats tracking
        self.episode_count: int = 0
        self.hit_count: int = 0
        self.last_reward: float = 0.0
        self.last_trajectory = None

    def _get_target_radius(self) -> float:
        """Get target radius based on stage config."""
        return self.config.get("target_radius", 0.5)

    def _randomize_target(self) -> None:
        """Spawn target at random position within configured range."""
        d_min, d_max = self.config["target_distance_range"]
        distance = self.np_random.uniform(d_min, d_max)

        # Random angle for placement (mostly forward, some lateral)
        angle = self.np_random.uniform(-np.pi / 4, np.pi / 4)

        # Target at roughly agent height ± some variation
        height = self.np_random.uniform(0.5, 3.0)

        target_pos = self.agent_position + np.array([
            distance * np.cos(angle),
            distance * np.sin(angle),
            height,
        ])

        # Target velocity (if moving)
        target_vel = np.zeros(3)
        if self.config["target_moving"]:
            sp_min, sp_max = self.config.get("target_speed_range", [1, 3])
            speed = self.np_random.uniform(sp_min, sp_max)
            direction = self.np_random.normal(size=3)
            direction[2] *= 0.3  # Less vertical movement
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            target_vel = direction * speed

        self.target = Target(
            id="target_0",
            position=target_pos,
            radius=self._get_target_radius(),
            velocity=target_vel,
            flag_color="red",
        )

    def _randomize_wind(self) -> None:
        """Set wind conditions based on stage config."""
        if self.config["wind_enabled"]:
            sp_min, sp_max = self.config.get("wind_speed_range", [0, 5])
            speed = self.np_random.uniform(sp_min, sp_max)
            direction = self.np_random.normal(size=3)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            self.wind = WindModel(
                direction=direction,
                speed=speed,
                gust_variance=speed * 0.1,
            )
        else:
            self.wind = WindModel()

    def _randomize_arrow(self) -> None:
        """Random arrow type."""
        self.arrow_type_index = self.np_random.integers(0, len(ARROW_TYPES))
        self.arrow_type = ARROW_TYPES[self.arrow_type_index]

    def _randomize_agent(self) -> None:
        """Randomize agent position if enabled."""
        if self.config["agent_moving"]:
            self.agent_position = np.array([
                self.np_random.uniform(-5, 5),
                self.np_random.uniform(-5, 5),
                1.0,  # Agent always at ground level + 1m
            ])
        else:
            self.agent_position = np.array([0.0, 0.0, 1.0])

        self.agent_forward = np.array([1.0, 0.0, 0.0])

    def _get_observation(self) -> np.ndarray:
        """Build the 22-element observation vector."""
        # Direction and distance to target
        to_target = self.target.position - self.agent_position
        distance = np.linalg.norm(to_target)
        elevation_diff = self.target.position[2] - self.agent_position[2]

        # Arrow type one-hot
        arrow_onehot = np.zeros(3)
        arrow_onehot[self.arrow_type_index] = 1.0

        obs = np.concatenate([
            self.agent_position,                         # 3
            self.agent_forward,                          # 3
            self.target.position,                        # 3
            self.target.velocity,                        # 3
            self.wind.direction,                         # 3
            [self.wind.speed],                           # 1
            arrow_onehot,                                # 3
            [self.draw_strength],                        # 1
            [distance],                                  # 1
            [elevation_diff],                            # 1
        ]).astype(np.float32)                            # Total: 22

        # Safety net: clamp NaN/inf to prevent training crash
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

        return obs

    def reset(self, seed=None, options=None):
        """Reset environment for new episode."""
        super().reset(seed=seed)

        self._randomize_agent()
        self._randomize_target()
        self._randomize_wind()
        self._randomize_arrow()
        self.draw_strength = 0.5  # Neutral starting draw

        obs = self._get_observation()
        info = {
            "target_distance": np.linalg.norm(
                self.target.position - self.agent_position
            ),
            "wind_speed": self.wind.speed,
            "arrow_type": self.arrow_type.name,
        }

        return obs, info

    def step(self, action: np.ndarray):
        """Execute one shot.

        Action mapping:
            action[0]: aim_pitch  [-1,1] → [-90°, 90°]
            action[1]: aim_yaw   [-1,1] → [-45°, 45°] relative to target bearing
            action[2]: draw_strength [-1,1] → [0.3, 1.0]
        """
        action = np.clip(action, -1.0, 1.0)

        # Map actions to physical values
        pitch = action[0] * np.radians(90)
        yaw_offset = action[1] * np.radians(45)

        # Compute bearing to target for yaw reference
        to_target = self.target.position - self.agent_position
        target_bearing = np.arctan2(to_target[1], to_target[0])
        yaw = target_bearing + yaw_offset

        # Map draw strength: [-1,1] → [0.3, 1.0]
        self.draw_strength = 0.3 + (action[2] + 1.0) * 0.35

        # Compute launch velocity
        velocity = compute_launch_velocity(
            pitch_angle=pitch,
            yaw_angle=yaw,
            draw_strength=self.draw_strength,
            arrow_type=self.arrow_type,
        )

        # Simulate trajectory (with seeded RNG for deterministic wind gusts)
        trajectory = simulate_trajectory(
            origin=self.agent_position.copy(),
            velocity_vector=velocity,
            arrow_type=self.arrow_type,
            wind=self.wind,
            rng=self.np_random,
        )
        self.last_trajectory = trajectory

        # Compute flight time from trajectory length
        flight_time = (len(trajectory) - 1) * 0.005  # dt=0.005 default

        # Move target to predicted position at impact time (for moving targets)
        if self.config["target_moving"] and flight_time > 0:
            predicted_target_pos = (
                self.target.position + self.target.velocity * flight_time
            )
        else:
            predicted_target_pos = self.target.position.copy()

        # Create a temporary target at predicted position for hit detection
        predicted_target = Target(
            id=self.target.id,
            position=predicted_target_pos,
            radius=self.target.radius,
            velocity=self.target.velocity,
            flag_color=self.target.flag_color,
        )

        # Check hit against predicted target position
        hit, hit_pos, dist_from_center = check_hit(trajectory, predicted_target)

        # Find closest approach for reward shaping
        closest_dist = min(
            np.linalg.norm(p - predicted_target_pos)
            for p, _ in trajectory
        )

        # Compute reward (continuous gradient signal via closest_approach)
        reward = compute_reward(
            hit, dist_from_center, predicted_target.radius,
            closest_approach=closest_dist,
        )

        self.last_reward = reward
        self.episode_count += 1
        if hit:
            self.hit_count += 1

        # Episode always ends after one shot
        terminated = True
        truncated = False

        obs = self._get_observation()
        info = {
            "hit": hit,
            "reward": reward,
            "distance_from_center": dist_from_center,
            "closest_approach": closest_dist,
            "target_distance": np.linalg.norm(
                self.target.position - self.agent_position
            ),
            "wind_speed": self.wind.speed,
            "arrow_type": self.arrow_type.name,
            "draw_strength": self.draw_strength,
        }

        return obs, reward, terminated, truncated, info

    @property
    def success_rate(self) -> float:
        """Rolling success rate."""
        if self.episode_count == 0:
            return 0.0
        return self.hit_count / self.episode_count


# ---------- Smoke Test ----------
if __name__ == "__main__":
    from rich.console import Console

    console = Console()
    console.print("\n[bold cyan]═══ LEAA Archery Environment Smoke Test ═══[/bold cyan]\n")

    # Test 1: Create environment and check spaces
    console.print("[bold]Test 1:[/bold] Environment creation and space validation")
    env = ArcheryEnv()
    console.print(f"  Observation space: {env.observation_space}")
    console.print(f"  Action space: {env.action_space}")
    assert env.observation_space.shape == (22,), f"Expected (22,), got {env.observation_space.shape}"
    assert env.action_space.shape == (3,), f"Expected (3,), got {env.action_space.shape}"
    console.print("  ✅ Spaces correct")

    # Test 2: Reset
    console.print("\n[bold]Test 2:[/bold] Environment reset")
    obs, info = env.reset(seed=42)
    console.print(f"  Obs shape: {obs.shape}, dtype: {obs.dtype}")
    console.print(f"  Info: {info}")
    assert obs.shape == (22,)
    assert obs.dtype == np.float32
    console.print("  ✅ Reset returns valid obs")

    # Test 3: Step with random action
    console.print("\n[bold]Test 3:[/bold] Step with random action")
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info2 = env.step(action)
    console.print(f"  Action: {action}")
    console.print(f"  Reward: {reward:.2f}")
    console.print(f"  Terminated: {terminated}")
    console.print(f"  Hit: {info2['hit']}")
    assert obs2.shape == (22,)
    assert terminated is True, "Episode should end after one shot"
    console.print("  ✅ Step returns valid obs/reward/done")

    # Test 4: Run 100 random episodes, check success rate
    console.print("\n[bold]Test 4:[/bold] 100 random episodes")
    env2 = ArcheryEnv(stage_config={"target_distance_range": [5, 10], "target_radius": 2.0})
    hits = 0
    total_reward = 0.0
    for i in range(100):
        obs, _ = env2.reset(seed=i)
        action = env2.action_space.sample()
        _, r, _, _, info = env2.step(action)
        total_reward += r
        if info["hit"]:
            hits += 1

    console.print(f"  Random agent hit rate: {hits}/100 = {hits}%")
    console.print(f"  Average reward: {total_reward/100:.2f}")
    console.print("  (Random agent should have low hit rate — that's expected)")
    console.print("  ✅ Environment runs 100 episodes without errors")

    # Test 5: Curriculum stage config
    console.print("\n[bold]Test 5:[/bold] Custom stage config (wind, moving)")
    stage_cfg = {
        "target_distance_range": [10, 30],
        "target_moving": True,
        "target_speed_range": [1, 3],
        "wind_enabled": True,
        "wind_speed_range": [2, 8],
        "target_radius": 1.0,
    }
    env3 = ArcheryEnv(stage_config=stage_cfg)
    obs, info = env3.reset(seed=99)
    console.print(f"  Wind speed: {info['wind_speed']:.1f} m/s")
    console.print(f"  Target distance: {info['target_distance']:.1f} m")
    console.print(f"  Target velocity: {env3.target.velocity}")
    assert info["wind_speed"] > 0, "Wind should be enabled"
    assert np.linalg.norm(env3.target.velocity) > 0, "Target should be moving"
    console.print("  ✅ Curriculum config applied correctly")

    console.print("\n[bold green]All archery environment tests passed![/bold green]\n")
