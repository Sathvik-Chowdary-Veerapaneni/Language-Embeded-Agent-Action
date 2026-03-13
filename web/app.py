"""
LEAA Web Demo — Flask + Three.js Archery Visualization

Flask backend serving the archery demo on port 3000.
Uses the existing physics engine and language grounding pipeline.
"""

import sys
import os
import math
import random

import numpy as np
from flask import Flask, render_template, jsonify, request

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from physics_engine.ballistics import (
    STANDARD_ARROW,
    HEAVY_ARROW,
    LIGHT_ARROW,
    ARROW_TYPES,
    WindModel,
    compute_launch_velocity,
    simulate_trajectory,
)
from physics_engine.collision import Target, check_hit
from rl_training.envs.scene_registry import SceneRegistry
from language_layer.grounding.pipeline import GroundingPipeline

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Scene state
# ---------------------------------------------------------------------------

registry = SceneRegistry()
pipeline = GroundingPipeline(use_llm_fallback=False)

AGENT_POS = np.array([0.0, 0.0, 1.5])  # archer position in physics coords
ARROW_ORIGIN = np.array([0.0, 0.0, 1.5])

TARGET_CONFIGS = [
    {"color": "red",    "x_range": (15, 25), "y_range": (-5, 5),   "moving": False},
    {"color": "blue",   "x_range": (25, 35), "y_range": (-8, 8),   "moving": True},
    {"color": "yellow", "x_range": (18, 28), "y_range": (-10, -3), "moving": False},
    {"color": "green",  "x_range": (30, 45), "y_range": (-6, 6),   "moving": True},
    {"color": "red",    "x_range": (35, 50), "y_range": (-4, 4),   "moving": False},
]


def _generate_scene():
    """Populate the registry with 5 targets."""
    registry.clear()
    for i, cfg in enumerate(TARGET_CONFIGS):
        x = random.uniform(*cfg["x_range"])
        y = random.uniform(*cfg["y_range"])
        z = 1.5  # target center height
        vel = np.zeros(3)
        if cfg["moving"]:
            vel[1] = random.choice([-1, 1]) * random.uniform(1.5, 3.0)
        registry.add_object(
            id=f"obj_{i}",
            position=[x, y, z],
            velocity=vel.tolist(),
            flag_color=cfg["color"],
            radius=0.5,
        )


_generate_scene()


def _physics_to_threejs(point):
    """Convert physics [x, y, z] to Three.js [x, z, y]."""
    return [float(point[0]), float(point[2]), float(point[1])]


def _scene_json():
    """Build JSON-safe scene representation with Three.js coords."""
    objects = []
    for obj in registry.get_all_active():
        objects.append({
            "id": obj.id,
            "position": _physics_to_threejs(obj.position),
            "velocity": _physics_to_threejs(obj.velocity),
            "flag_color": obj.flag_color,
            "radius": obj.radius,
            "is_moving": float(np.linalg.norm(obj.velocity)) > 0.1,
        })
    return {"objects": objects}


def _heuristic_aim(target_pos):
    """Compute pitch and yaw to hit a target using iterative search."""
    dx = target_pos[0] - ARROW_ORIGIN[0]
    dy = target_pos[1] - ARROW_ORIGIN[1]
    dz = target_pos[2] - ARROW_ORIGIN[2]
    horiz_dist = math.sqrt(dx ** 2 + dy ** 2)
    yaw = math.atan2(dy, dx)

    best_pitch = 0.1
    best_miss = float("inf")

    for pitch_deg in range(1, 60):
        pitch = math.radians(pitch_deg)
        vel = compute_launch_velocity(pitch, yaw, 1.0, STANDARD_ARROW)
        traj = simulate_trajectory(ARROW_ORIGIN.copy(), vel, STANDARD_ARROW, WindModel(), dt=0.01, max_time=5.0)

        for pos, _ in traj:
            dist = np.linalg.norm(pos - target_pos)
            if dist < best_miss:
                best_miss = dist
                best_pitch = pitch

    return best_pitch, yaw


def _sample_trajectory(traj, max_points=120):
    """Downsample trajectory to at most max_points."""
    if len(traj) <= max_points:
        return traj
    step = max(1, len(traj) // max_points)
    sampled = traj[::step]
    
    # Check if the very last item of the original trajectory is the same object 
    # reference as the very last item of the sampled trajectory. 
    # If not, append it to guarantee we reach the target point.
    if sampled[-1] is not traj[-1]:
        sampled.append(traj[-1])
        
    return sampled


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/scene")
def get_scene():
    return jsonify(_scene_json())


@app.route("/api/fire", methods=["POST"])
def fire():
    data = request.get_json(force=True)
    command = data.get("command", "shoot the closest target")

    # Ground the command
    resolved = pipeline.ground(command, registry, AGENT_POS)
    if resolved is None:
        return jsonify({"error": "Could not resolve target", "hit": False}), 200

    target_obj = resolved.target_obj
    target_pos = target_obj.position.copy()

    # Heuristic aim
    pitch, yaw = _heuristic_aim(target_pos)
    vel = compute_launch_velocity(pitch, yaw, 1.0, STANDARD_ARROW)

    # Simulate
    traj = simulate_trajectory(
        ARROW_ORIGIN.copy(), vel, STANDARD_ARROW, WindModel(), dt=0.005, max_time=5.0,
    )

    # Check hit
    collision_target = Target(
        id=target_obj.id,
        position=target_obj.position.copy(),
        radius=target_obj.radius,
    )
    hit, hit_pos, dist_from_center = check_hit(traj, collision_target)

    # Sample trajectory for frontend
    sampled = _sample_trajectory(traj, max_points=120)
    trajectory_points = [_physics_to_threejs(pos) for pos, _ in sampled]

    # Distance from archer to target
    target_distance = float(np.linalg.norm(target_pos - ARROW_ORIGIN))

    if hit:
        result_text = f"Hit {target_obj.flag_color} target ({target_obj.id}) at {target_distance:.1f}m!"
    else:
        miss_dist = dist_from_center if dist_from_center is not None else 999
        result_text = f"Missed {target_obj.flag_color} target by {miss_dist:.2f}m"

    return jsonify({
        "target_id": target_obj.id,
        "hit": hit,
        "trajectory_points": trajectory_points,
        "target_position": _physics_to_threejs(target_pos),
        "result_text": result_text,
        "distance": target_distance,
        "flag_color": target_obj.flag_color,
    })


@app.route("/api/randomize", methods=["POST"])
def randomize():
    _generate_scene()
    return jsonify(_scene_json())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
