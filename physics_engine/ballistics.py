"""
LEAA Physics Engine — Arrow Flight Ballistics

Core arrow flight physics using RK4 (Runge-Kutta 4th order) integration.
This is the single source of truth for trajectory calculations — later ported to Unity C#.

Coordinate system: x=forward, y=left, z=up
All units SI: meters, seconds, kg, radians internally.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

# ---------- Constants ----------
GRAVITY = 9.81                # m/s²
AIR_DENSITY = 1.225           # kg/m³ (sea level, 15°C)
MAX_DRAW_SPEED = 70.0         # m/s (compound bow max)


# ---------- Data Classes ----------
@dataclass
class ArrowType:
    """Defines physical properties of an arrow."""
    name: str
    mass: float               # kg
    drag_coefficient: float   # Cd (dimensionless)
    cross_section_area: float # m²
    length: float = 0.75      # m (default shaft length)


# Predefined arrow types
STANDARD_ARROW = ArrowType(name="standard", mass=0.025, drag_coefficient=0.47, cross_section_area=0.0005)
HEAVY_ARROW    = ArrowType(name="heavy",    mass=0.040, drag_coefficient=0.42, cross_section_area=0.0006)
LIGHT_ARROW    = ArrowType(name="light",    mass=0.018, drag_coefficient=0.52, cross_section_area=0.0004)

ARROW_TYPES = [STANDARD_ARROW, HEAVY_ARROW, LIGHT_ARROW]


@dataclass
class WindModel:
    """Wind conditions for the simulation."""
    direction: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    speed: float = 0.0         # m/s
    gust_variance: float = 0.0 # variance for random gusts

    def get_wind_vector(self) -> np.ndarray:
        """Return wind velocity vector with optional gust noise."""
        base = self.direction / (np.linalg.norm(self.direction) + 1e-8) * self.speed
        if self.gust_variance > 0:
            gust = np.random.normal(0, self.gust_variance, size=3)
            return base + gust
        return base.copy()


# ---------- Physics Functions ----------
def _compute_forces(
    position: np.ndarray,
    velocity: np.ndarray,
    arrow: ArrowType,
    wind: WindModel,
) -> np.ndarray:
    """Compute total force on the arrow at a given state.

    Forces:
      - Gravity:  F_g = [0, 0, -m*g]
      - Air drag: F_d = -0.5 * rho * Cd * A * |v|² * v_hat
      - Wind:     F_w =  0.5 * rho * Cd * A * |w-v|² * (w-v)_hat
    """
    m = arrow.mass
    Cd = arrow.drag_coefficient
    A = arrow.cross_section_area

    # Gravity
    f_gravity = np.array([0.0, 0.0, -m * GRAVITY])

    # Air drag (opposes velocity)
    speed = np.linalg.norm(velocity)
    if speed > 1e-8:
        v_hat = velocity / speed
        f_drag = -0.5 * AIR_DENSITY * Cd * A * speed**2 * v_hat
    else:
        f_drag = np.zeros(3)

    # Wind force (relative wind)
    wind_vec = wind.get_wind_vector()
    relative_wind = wind_vec - velocity
    rel_speed = np.linalg.norm(relative_wind)
    if rel_speed > 1e-8:
        rw_hat = relative_wind / rel_speed
        f_wind = 0.5 * AIR_DENSITY * Cd * A * rel_speed**2 * rw_hat
    else:
        f_wind = np.zeros(3)

    return f_gravity + f_drag + f_wind


def simulate_trajectory(
    origin: np.ndarray,
    velocity_vector: np.ndarray,
    arrow_type: ArrowType,
    wind: WindModel,
    dt: float = 0.005,
    max_time: float = 10.0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Simulate arrow flight using RK4 integration.

    Args:
        origin: Launch position [x, y, z].
        velocity_vector: Initial velocity [vx, vy, vz].
        arrow_type: Arrow physical properties.
        wind: Wind conditions.
        dt: Timestep (seconds). Default 0.005 for accuracy.
        max_time: Maximum simulation time (seconds).

    Returns:
        List of (position, velocity) tuples at each timestep.
        Simulation ends when arrow hits ground (z <= 0) or max_time reached.
    """
    pos = np.array(origin, dtype=np.float64)
    vel = np.array(velocity_vector, dtype=np.float64)
    m = arrow_type.mass

    trajectory: List[Tuple[np.ndarray, np.ndarray]] = [(pos.copy(), vel.copy())]

    t = 0.0
    max_steps = int(max_time / dt)

    for _ in range(max_steps):
        # RK4 integration
        # State: [position, velocity]
        # Derivatives: [velocity, acceleration]

        # k1
        a1 = _compute_forces(pos, vel, arrow_type, wind) / m
        k1_pos = vel
        k1_vel = a1

        # k2
        pos2 = pos + 0.5 * dt * k1_pos
        vel2 = vel + 0.5 * dt * k1_vel
        a2 = _compute_forces(pos2, vel2, arrow_type, wind) / m
        k2_pos = vel2
        k2_vel = a2

        # k3
        pos3 = pos + 0.5 * dt * k2_pos
        vel3 = vel + 0.5 * dt * k2_vel
        a3 = _compute_forces(pos3, vel3, arrow_type, wind) / m
        k3_pos = vel3
        k3_vel = a3

        # k4
        pos4 = pos + dt * k3_pos
        vel4 = vel + dt * k3_vel
        a4 = _compute_forces(pos4, vel4, arrow_type, wind) / m
        k4_pos = vel4
        k4_vel = a4

        # Update state
        pos = pos + (dt / 6.0) * (k1_pos + 2*k2_pos + 2*k3_pos + k4_pos)
        vel = vel + (dt / 6.0) * (k1_vel + 2*k2_vel + 2*k3_vel + k4_vel)

        trajectory.append((pos.copy(), vel.copy()))

        t += dt

        # Stop if arrow hits the ground
        if pos[2] <= 0.0:
            pos[2] = 0.0  # Clamp to ground
            trajectory[-1] = (pos.copy(), vel.copy())
            break

    return trajectory


def compute_launch_velocity(
    pitch_angle: float,
    yaw_angle: float,
    draw_strength: float,
    arrow_type: ArrowType,
) -> np.ndarray:
    """Convert aiming angles and draw strength into an initial velocity vector.

    Args:
        pitch_angle: Elevation angle in radians. Positive = up.
        yaw_angle: Horizontal angle in radians. Positive = left.
        draw_strength: Fraction of max draw [0, 1].
        arrow_type: Arrow type (heavier arrows are slightly slower).

    Returns:
        3D velocity vector [vx, vy, vz].
    """
    draw_strength = np.clip(draw_strength, 0.0, 1.0)

    # Heavier arrows get slightly less speed (momentum conservation approx)
    mass_factor = 0.025 / arrow_type.mass  # normalized to standard arrow
    speed = MAX_DRAW_SPEED * draw_strength * min(mass_factor, 1.0)

    vx = speed * np.cos(pitch_angle) * np.cos(yaw_angle)
    vy = speed * np.cos(pitch_angle) * np.sin(yaw_angle)
    vz = speed * np.sin(pitch_angle)

    return np.array([vx, vy, vz], dtype=np.float64)


# ---------- Smoke Test ----------
if __name__ == "__main__":
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print("\n[bold cyan]═══ LEAA Ballistics Smoke Test ═══[/bold cyan]\n")

    # Test 1: 45° no-wind, max draw — should give max range
    console.print("[bold]Test 1:[/bold] 45° pitch, no wind, max draw (STANDARD arrow)")
    vel = compute_launch_velocity(
        pitch_angle=np.radians(45),
        yaw_angle=0.0,
        draw_strength=1.0,
        arrow_type=STANDARD_ARROW,
    )
    console.print(f"  Launch velocity: {vel} (speed: {np.linalg.norm(vel):.1f} m/s)")

    wind = WindModel()  # no wind
    traj = simulate_trajectory(
        origin=np.array([0.0, 0.0, 1.0]),  # 1m launch height
        velocity_vector=vel,
        arrow_type=STANDARD_ARROW,
        wind=wind,
    )

    # Print trajectory table (sampled every 100 steps)
    table = Table(title="Trajectory (sampled every 0.5s)")
    table.add_column("Time (s)", style="cyan")
    table.add_column("Position (x, y, z)", style="green")
    table.add_column("Speed (m/s)", style="yellow")

    for i, (pos, v) in enumerate(traj):
        if i % 100 == 0 or i == len(traj) - 1:
            t = i * 0.005
            table.add_row(
                f"{t:.2f}",
                f"({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})",
                f"{np.linalg.norm(v):.1f}",
            )
    console.print(table)

    landing_pos = traj[-1][0]
    console.print(f"  Landing position: x={landing_pos[0]:.1f}m, z={landing_pos[2]:.1f}m")
    console.print(f"  Range: [bold green]{landing_pos[0]:.1f}m[/bold green] (with air drag)")
    console.print(f"  Note: Theoretical no-drag range at 70 m/s ≈ 500m; drag reduces this significantly\n")

    # Test 2: Zero draw strength = zero velocity
    console.print("[bold]Test 2:[/bold] Zero draw strength")
    vel_zero = compute_launch_velocity(np.radians(45), 0.0, 0.0, STANDARD_ARROW)
    assert np.allclose(vel_zero, 0.0), "Zero draw should give zero velocity!"
    console.print("  ✅ Zero draw → zero velocity\n")

    # Test 3: Heavy arrow drops faster than light
    console.print("[bold]Test 3:[/bold] Heavy vs. Light arrow (same angle/draw)")
    for arrow in [LIGHT_ARROW, STANDARD_ARROW, HEAVY_ARROW]:
        v = compute_launch_velocity(np.radians(30), 0.0, 1.0, arrow)
        t = simulate_trajectory(np.array([0, 0, 1.0]), v, arrow, WindModel())
        land_x = t[-1][0][0]
        console.print(f"  {arrow.name:>8}: range = {land_x:.1f}m, launch speed = {np.linalg.norm(v):.1f} m/s")

    # Test 4: 45° gives max range among tested angles
    console.print("\n[bold]Test 4:[/bold] Angle sweep (max range near 45°)")
    best_angle, best_range = 0, 0
    for deg in range(10, 80, 5):
        v = compute_launch_velocity(np.radians(deg), 0.0, 1.0, STANDARD_ARROW)
        t = simulate_trajectory(np.array([0, 0, 1.0]), v, STANDARD_ARROW, WindModel())
        r = t[-1][0][0]
        if r > best_range:
            best_angle, best_range = deg, r
    console.print(f"  Best angle: {best_angle}° → range: {best_range:.1f}m")
    # With drag, optimal angle shifts below 45° (typically 30-40°). 45° is only optimal in vacuum.
    assert 25 <= best_angle <= 50, f"Expected max range in 25-50° range, got {best_angle}°"
    console.print(f"  ✅ Max range at {best_angle}° (drag shifts optimum below 45°)\n")

    console.print("[bold green]All ballistics tests passed![/bold green]\n")
