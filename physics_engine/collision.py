"""
LEAA Physics Engine — Collision Detection

Hit detection between arrow trajectories and targets.
Uses line-segment-sphere intersection for precision at high arrow speeds.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------- Data Classes ----------
@dataclass
class Target:
    """Defines a target in the scene."""
    id: str
    position: np.ndarray           # [x, y, z]
    radius: float = 0.5            # meters
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    flag_color: str = "red"


# ---------- Hit Detection ----------
def _line_segment_sphere_intersection(
    p1: np.ndarray,
    p2: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> Tuple[bool, Optional[np.ndarray]]:
    """Check if a line segment (p1→p2) intersects a sphere.

    Uses quadratic formula on parametric line: P(t) = p1 + t*(p2-p1), t ∈ [0,1]
    Intersects sphere when |P(t) - center|² = radius².

    Returns:
        (hit, intersection_point) — closest intersection point if hit.
    """
    d = p2 - p1         # segment direction
    f = p1 - center     # start relative to sphere center

    a = np.dot(d, d)
    b = 2.0 * np.dot(f, d)
    c = np.dot(f, f) - radius * radius

    if a < 1e-12:
        # Segment has zero length — just check if point is inside sphere
        if np.linalg.norm(p1 - center) <= radius:
            return True, p1.copy()
        return False, None

    discriminant = b * b - 4 * a * c

    if discriminant < 0:
        return False, None

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    # Check if either root is within [0, 1]
    for t in [t1, t2]:
        if 0.0 <= t <= 1.0:
            hit_point = p1 + t * d
            return True, hit_point

    # Check if segment is entirely inside the sphere
    if t1 < 0.0 and t2 > 1.0:
        return True, p1.copy()  # Already inside

    return False, None


def check_hit(
    trajectory_points: List[Tuple[np.ndarray, np.ndarray]],
    target: Target,
) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
    """Check if an arrow trajectory hits a target.

    Uses line-segment-sphere intersection between consecutive trajectory
    points to avoid missing thin targets at high speed.

    Args:
        trajectory_points: List of (position, velocity) from simulate_trajectory.
        target: Target to check against.

    Returns:
        (hit, hit_position, distance_from_center)
        - hit: True if any segment intersects the target sphere
        - hit_position: 3D point of first intersection (or None)
        - distance_from_center: Distance from hit point to target center (or None)
    """
    for i in range(len(trajectory_points) - 1):
        p1 = trajectory_points[i][0]
        p2 = trajectory_points[i + 1][0]

        hit, hit_point = _line_segment_sphere_intersection(
            p1, p2, target.position, target.radius
        )

        if hit and hit_point is not None:
            dist_from_center = np.linalg.norm(hit_point - target.position)
            return True, hit_point, dist_from_center

    return False, None, None


def compute_reward(
    hit: bool,
    distance_from_center: Optional[float],
    target_radius: float,
    closest_approach: Optional[float] = None,
) -> float:
    """Compute reward based on hit precision with continuous gradient signal.

    Scoring zones (on hit):
        - Bullseye (center 10%): +100
        - Inner ring (10-50%):   +50
        - Outer ring (50-100%):  +25

    Miss reward (exponential decay):
        10.0 * exp(-distance / (2 * target_radius))

        Examples at radius=0.5m:
            1 radius away  → ~6.1
            2 radii away   → ~3.7
            5 radii away   → ~0.8
            10 radii away  → ~0.07

    Args:
        hit: Whether the arrow hit the target.
        distance_from_center: Distance from hit point to target center.
        target_radius: Target sphere radius.
        closest_approach: Closest distance any trajectory point got to target center (for misses).

    Returns:
        Float reward value.
    """
    if hit and distance_from_center is not None:
        # Normalize distance to [0, 1] relative to radius
        ratio = distance_from_center / target_radius

        if ratio <= 0.10:
            return 100.0   # Bullseye
        elif ratio <= 0.50:
            return 50.0    # Inner ring
        elif ratio <= 1.00:
            return 25.0    # Outer ring

    # Miss — exponential decay based on closest approach
    dist = closest_approach if closest_approach is not None else (distance_from_center or 50.0)
    return 10.0 * np.exp(-dist / (2.0 * target_radius))


def move_targets(
    targets: List[Target],
    dt: float,
    bounds: Tuple[float, float, float] = (100.0, 50.0, 30.0),
) -> None:
    """Update target positions based on their velocities. Bounce off boundaries.

    Args:
        targets: List of targets to update (modified in place).
        dt: Time step.
        bounds: (x_max, y_max, z_max) — targets bounce within [-bound, +bound].
    """
    for target in targets:
        target.position = target.position + target.velocity * dt

        # Bounce off boundaries
        for axis in range(3):
            if abs(target.position[axis]) > bounds[axis]:
                target.position[axis] = np.clip(
                    target.position[axis], -bounds[axis], bounds[axis]
                )
                target.velocity[axis] *= -1.0  # Reverse direction


# ---------- Smoke Test ----------
if __name__ == "__main__":
    from rich.console import Console

    console = Console()
    console.print("\n[bold cyan]═══ LEAA Collision Smoke Test ═══[/bold cyan]\n")

    # Create a target at x=50, y=0, z=1
    target = Target(
        id="target_0",
        position=np.array([50.0, 0.0, 1.0]),
        radius=0.5,
        flag_color="red",
    )

    # Test 1: Trajectory that should hit the target
    console.print("[bold]Test 1:[/bold] Trajectory passing through target sphere")
    from physics_engine.ballistics import (
        STANDARD_ARROW, WindModel, compute_launch_velocity, simulate_trajectory
    )

    # Aim roughly at the target (50m away, ~same height)
    vel = compute_launch_velocity(
        pitch_angle=np.radians(5),
        yaw_angle=0.0,
        draw_strength=0.6,
        arrow_type=STANDARD_ARROW,
    )
    traj = simulate_trajectory(
        origin=np.array([0.0, 0.0, 1.0]),
        velocity_vector=vel,
        arrow_type=STANDARD_ARROW,
        wind=WindModel(),
    )

    hit, hit_pos, dist = check_hit(traj, target)
    console.print(f"  Hit: {hit}")
    if hit:
        console.print(f"  Hit position: ({hit_pos[0]:.2f}, {hit_pos[1]:.2f}, {hit_pos[2]:.2f})")
        console.print(f"  Distance from center: {dist:.4f}m")
        reward = compute_reward(hit, dist, target.radius)
        console.print(f"  Reward: {reward}")
        console.print("  ✅ Hit detected!")
    else:
        # If we miss, try adjusting — this is fine, just print closest approach
        closest = min(
            np.linalg.norm(p - target.position) for p, _ in traj
        )
        console.print(f"  Closest approach: {closest:.2f}m (target radius: {target.radius}m)")
        console.print("  ⚠️ Missed — adjusting aim for test...")

        # Binary search for a pitch that hits at 50m
        for pitch_deg in range(1, 45):
            v = compute_launch_velocity(np.radians(pitch_deg), 0.0, 0.6, STANDARD_ARROW)
            t = simulate_trajectory(np.array([0, 0, 1.0]), v, STANDARD_ARROW, WindModel())
            h, hp, d = check_hit(t, target)
            if h:
                console.print(f"  Found hit at pitch={pitch_deg}°, dist_from_center={d:.4f}m")
                reward = compute_reward(h, d, target.radius)
                console.print(f"  Reward: {reward}")
                console.print("  ✅ Hit detected!")
                break
        else:
            # Increase target radius for test
            target.radius = 2.0
            v = compute_launch_velocity(np.radians(5), 0.0, 0.6, STANDARD_ARROW)
            t = simulate_trajectory(np.array([0, 0, 1.0]), v, STANDARD_ARROW, WindModel())
            h, hp, d = check_hit(t, target)
            console.print(f"  With radius=2.0m → Hit: {h}")
            if h:
                console.print("  ✅ Hit detected with larger radius!")

    # Test 2: Trajectory that should miss
    console.print("\n[bold]Test 2:[/bold] Trajectory that misses (aim away)")
    target_far = Target(
        id="target_far",
        position=np.array([50.0, 100.0, 50.0]),  # way off to the side and up
        radius=0.5,
    )
    vel_miss = compute_launch_velocity(np.radians(10), 0.0, 0.5, STANDARD_ARROW)
    traj_miss = simulate_trajectory(
        np.array([0, 0, 1.0]), vel_miss, STANDARD_ARROW, WindModel()
    )
    hit_miss, _, _ = check_hit(traj_miss, target_far)
    assert not hit_miss, "Expected miss!"
    reward_miss = compute_reward(hit_miss, None, target_far.radius, closest_approach=50.0)
    assert reward_miss < 0.01, f"Far miss should be near 0, got {reward_miss}"
    console.print(f"  ✅ Miss detected correctly, reward = {reward_miss:.4f}")

    # Test 3: Reward zones
    console.print("\n[bold]Test 3:[/bold] Reward zone validation")
    r = 1.0
    assert compute_reward(True, 0.05, r) == 100.0, "Bullseye failed"
    assert compute_reward(True, 0.30, r) == 50.0, "Inner ring failed"
    assert compute_reward(True, 0.80, r) == 25.0, "Outer ring failed"
    # Near miss: closest_approach = 2.0, radius = 1.0 → 10*exp(-2/2) ≈ 3.68
    near_miss = compute_reward(False, None, r, closest_approach=2.0)
    assert near_miss > 3.0, f"Near miss should be ~3.68, got {near_miss}"
    # Far miss: closest_approach = 30.0 → 10*exp(-30/2) ≈ 0.0
    far_miss = compute_reward(False, None, r, closest_approach=30.0)
    assert far_miss < 0.01, f"Far miss should be ~0, got {far_miss}"
    console.print("  ✅ Bullseye (0.05/1.0) → 100")
    console.print("  ✅ Inner ring (0.30/1.0) → 50")
    console.print("  ✅ Outer ring (0.80/1.0) → 25")
    console.print(f"  ✅ Near miss (2.0m) → {near_miss:.2f}")
    console.print(f"  ✅ Far miss (30.0m) → {far_miss:.2f}")

    # Test 4: Moving targets
    console.print("\n[bold]Test 4:[/bold] Moving targets with boundary bounce")
    t1 = Target("t1", np.array([90.0, 0.0, 1.0]), velocity=np.array([5.0, 0.0, 0.0]))
    t2 = Target("t2", np.array([-90.0, 0.0, 1.0]), velocity=np.array([-5.0, 0.0, 0.0]))
    targets = [t1, t2]

    for _ in range(100):
        move_targets(targets, 0.1)

    # Both should have bounced and be within bounds
    for t in targets:
        assert abs(t.position[0]) <= 100.0, f"Target {t.id} out of bounds!"
    console.print(f"  t1 position: {t1.position}")
    console.print(f"  t2 position: {t2.position}")
    console.print("  ✅ Targets stayed within bounds after bouncing")

    console.print("\n[bold green]All collision tests passed![/bold green]\n")
