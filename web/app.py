"""
LEAA Web Demo — Flask + Three.js Archery Visualization

Flask backend serving the archery demo on port 3000.
Uses the existing physics engine, language grounding pipeline,
and trained RL agent for intelligent aiming.
"""

import sys
import os
import math
import json
import random
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, render_template, jsonify, request

# ---------------------------------------------------------------------------
# Logging setup — separate files for heuristic and RL analysis
# ---------------------------------------------------------------------------
LOG_DIR = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "web", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

def _make_logger(name, filename):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(os.path.join(LOG_DIR, filename), mode="a")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger

heuristic_log = _make_logger("heuristic", "heuristic_shots.log")
rl_log = _make_logger("rl", "rl_shots.log")

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

# RL model imports
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.monitor import Monitor
    from rl_training.envs.archery_env import ArcheryEnv
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

app = Flask(__name__)


from flask.json.provider import DefaultJSONProvider

class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)

# ---------------------------------------------------------------------------
# RL Agent loader
# ---------------------------------------------------------------------------

rl_model = None
rl_vec_normalize = None
rl_loaded = False

def _load_rl_model():
    """Load the best trained RL model and VecNormalize stats."""
    global rl_model, rl_vec_normalize, rl_loaded

    if not RL_AVAILABLE:
        print("  [RL] stable_baselines3 not installed — RL mode disabled")
        return

    # Search for best checkpoint (prefer moving_slow which achieved 70%+,
    # wind_best only reached ~4.6% so it's worse for inference)
    checkpoint_search = [
        ("cloud_checkpoints/moving_slow/moving_slow_best.zip", "cloud_checkpoints/moving_slow/vecnormalize_moving_slow_best.pkl"),
        ("cloud_checkpoints/wind/wind_best.zip", "cloud_checkpoints/wind/vecnormalize_wind_best.pkl"),
        ("cloud_checkpoints/static_far/static_far_best.zip", "cloud_checkpoints/static_far/vecnormalize_static_far_best.pkl"),
        ("rl_training/checkpoints/moving_slow_best.zip", "rl_training/checkpoints/vecnormalize_moving_slow_best.pkl"),
    ]

    model_path = None
    vecnorm_path = None
    for mp, vp in checkpoint_search:
        full_mp = os.path.join(PROJECT_ROOT, mp)
        full_vp = os.path.join(PROJECT_ROOT, vp)
        if os.path.exists(full_mp):
            model_path = full_mp
            vecnorm_path = full_vp if os.path.exists(full_vp) else None
            break

    if model_path is None:
        print("  [RL] No trained model found — RL mode disabled")
        return

    try:
        rl_model = PPO.load(model_path)
        print(f"  [RL] Loaded model: {os.path.basename(model_path)}")

        if vecnorm_path:
            dummy_env = DummyVecEnv([lambda: Monitor(ArcheryEnv())])
            rl_vec_normalize = VecNormalize.load(vecnorm_path, dummy_env)
            rl_vec_normalize.training = False
            rl_vec_normalize.norm_reward = False
            print(f"  [RL] Loaded VecNormalize: {os.path.basename(vecnorm_path)}")

        rl_loaded = True
        print("  [RL] RL agent ready!")
    except Exception as e:
        print(f"  [RL] Failed to load model: {e}")
        rl_loaded = False


def _build_rl_observation(target_obj, wind, arrow_type_index=0):
    """Build the 22-element observation vector matching ArcheryEnv._get_observation().

    Observation layout:
        agent_position (3) + agent_forward (3) + target_position (3) +
        target_velocity (3) + wind_direction (3) + wind_speed (1) +
        arrow_onehot (3) + pitch_hint (1) + distance (1) + elevation_diff (1)
    """
    agent_pos = AGENT_POS.copy()
    agent_forward = np.array([1.0, 0.0, 0.0])
    target_pos = np.array(target_obj.position, dtype=float)
    target_vel = np.array(target_obj.velocity, dtype=float)

    to_target = target_pos - agent_pos
    distance = np.linalg.norm(to_target)
    elevation_diff = target_pos[2] - agent_pos[2]

    # Arrow type one-hot
    arrow_onehot = np.zeros(3)
    arrow_onehot[arrow_type_index] = 1.0

    # Physics-based pitch hint (matches ArcheryEnv)
    h_dist = max(np.sqrt(to_target[0] ** 2 + to_target[1] ** 2), 1e-8)
    v_eff = 70.0 * 0.65
    gravity_comp = 0.5 * 9.81 * (h_dist / v_eff) ** 2
    pitch_hint_rad = np.arctan2(elevation_diff + gravity_comp, h_dist)
    pitch_hint_norm = float(np.clip(pitch_hint_rad / np.radians(30), -1.0, 1.0))

    obs = np.concatenate([
        agent_pos,            # 3
        agent_forward,        # 3
        target_pos,           # 3
        target_vel,           # 3
        wind.direction,       # 3
        [wind.speed],         # 1
        arrow_onehot,         # 3
        [pitch_hint_norm],    # 1
        [distance],           # 1
        [elevation_diff],     # 1
    ]).astype(np.float32)     # Total: 22

    obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
    return obs


def _rl_aim(target_obj, wind, arrow_type=STANDARD_ARROW, arrow_type_index=0):
    """Use the trained RL agent to compute pitch, yaw, and draw_strength."""
    obs = _build_rl_observation(target_obj, wind, arrow_type_index)

    # Normalize observation if VecNormalize is available
    if rl_vec_normalize is not None:
        obs = rl_vec_normalize.normalize_obs(obs)

    action, _ = rl_model.predict(obs, deterministic=True)
    action = np.clip(action, -1.0, 1.0)

    # Map actions (matches ArcheryEnv.step)
    pitch = action[0] * np.radians(30)

    # Yaw: relative to target bearing
    to_target = np.array(target_obj.position, dtype=float) - AGENT_POS
    target_bearing = np.arctan2(to_target[1], to_target[0])
    yaw_offset = action[1] * np.radians(45)
    yaw = target_bearing + yaw_offset

    # Draw strength: [-1,1] → [0.3, 1.0]
    draw_strength = 0.3 + (action[2] + 1.0) * 0.35

    return pitch, yaw, draw_strength


# ---------------------------------------------------------------------------
# Scene state
# ---------------------------------------------------------------------------

registry = SceneRegistry()
pipeline = GroundingPipeline(use_llm_fallback=False)

AGENT_POS = np.array([0.0, 0.0, 1.5])  # archer bow height in scene
ARROW_ORIGIN = np.array([0.0, 0.0, 1.5])

# Load RL model at startup (after AGENT_POS is defined)
_load_rl_model()

TARGET_CONFIGS = [
    {"color": "red",    "x_range": (15, 18), "y_range": (-8, -5),  "moving": False, "shape": "bullseye", "severity": 1, "radius": 0.7},
    {"color": "blue",   "x_range": (22, 26), "y_range": (-2, 2),   "moving": False, "shape": "bullseye", "severity": 2, "radius": 0.7},
    {"color": "yellow", "x_range": (28, 33), "y_range": (5, 8),    "moving": True,  "shape": "bullseye", "severity": 3, "radius": 0.7},
    {"color": "green",  "x_range": (35, 40), "y_range": (-6, -3),  "moving": True,  "shape": "bullseye", "severity": 4, "radius": 0.7},
    {"color": "white",  "x_range": (42, 48), "y_range": (2, 5),    "moving": False, "shape": "bullseye", "severity": 5, "radius": 0.7},
]


def _generate_scene():
    """Populate the registry with archery targets matching RL training."""
    registry.clear()
    for i, cfg in enumerate(TARGET_CONFIGS):
        x = random.uniform(*cfg["x_range"])
        y = random.uniform(*cfg["y_range"])
        z = 1.35  # board center height on the bullseye stand
        vel = np.zeros(3)
        if cfg["moving"]:
            vel[1] = random.choice([-1, 1]) * random.uniform(1.0, 3.0)
        registry.add_object(
            id=f"target_{i}",
            position=[x, y, z],
            velocity=vel.tolist(),
            flag_color=cfg["color"],
            radius=cfg["radius"],
        )


_generate_scene()


def _physics_to_threejs(point):
    """Convert physics [x, y, z] to Three.js [x, z, y]."""
    return [float(point[0]), float(point[2]), float(point[1])]


def _scene_json():
    """Build JSON-safe scene representation with Three.js coords."""
    objects = []
    for idx, obj in enumerate(registry.get_all_active()):
        cfg = TARGET_CONFIGS[idx] if idx < len(TARGET_CONFIGS) else {}
        objects.append({
            "id": obj.id,
            "position": _physics_to_threejs(obj.position),
            "velocity": _physics_to_threejs(obj.velocity),
            "flag_color": obj.flag_color,
            "radius": obj.radius,
            "is_moving": float(np.linalg.norm(obj.velocity)) > 0.1,
            "shape": "bullseye",
            "severity": cfg.get("severity", 1),
        })
    return {"objects": objects}


def _heuristic_aim(target_pos, wind=None):
    """Compute pitch and yaw to hit a target using iterative search."""
    if wind is None:
        wind = WindModel()
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
        traj = simulate_trajectory(ARROW_ORIGIN.copy(), vel, STANDARD_ARROW, wind, dt=0.01, max_time=5.0)

        for pos, _ in traj:
            dist = np.linalg.norm(pos - target_pos)
            if dist < best_miss:
                best_miss = dist
                best_pitch = pitch

    return best_pitch, yaw


def _aim_from_direction(aim_dir, wind=None):
    """Compute yaw from mouse aim, then search for optimal pitch.

    Yaw = where the player points (mouse-controlled).
    Pitch = auto-searched to maximize accuracy along that yaw line.
    """
    if wind is None:
        wind = WindModel()
    # Three.js [x, y, z] → physics [x, z, y]
    dx = aim_dir["x"]
    dy = aim_dir["z"]  # Three.js Z → physics Y
    dz = aim_dir["y"]  # Three.js Y → physics Z

    yaw = math.atan2(dy, dx)

    # Find the closest target along this yaw direction to aim at
    best_target_pos = None
    best_dot = -1.0
    aim_horiz = np.array([math.cos(yaw), math.sin(yaw)])
    for obj in registry.get_all_active():
        to_obj = obj.position[:2] - ARROW_ORIGIN[:2]
        dist = np.linalg.norm(to_obj)
        if dist < 1.0:
            continue
        obj_dir = to_obj / dist
        dot = np.dot(aim_horiz, obj_dir)
        if dot > best_dot:
            best_dot = dot
            best_target_pos = obj.position.copy()

    # If we found a target roughly in this direction, search for best pitch
    if best_target_pos is not None and best_dot > 0.7:
        best_pitch = 0.1
        best_miss = float("inf")
        for pitch_deg in range(1, 45):
            pitch = math.radians(pitch_deg)
            vel = compute_launch_velocity(pitch, yaw, 1.0, STANDARD_ARROW)
            traj = simulate_trajectory(ARROW_ORIGIN.copy(), vel, STANDARD_ARROW, wind, dt=0.01, max_time=5.0)
            for pos, _ in traj:
                dist = np.linalg.norm(pos - best_target_pos)
                if dist < best_miss:
                    best_miss = dist
                    best_pitch = pitch
        return best_pitch, yaw

    # No target in this direction — use raw aim with basic gravity comp
    horiz = math.sqrt(dx ** 2 + dy ** 2)
    pitch = math.atan2(dz, horiz) if horiz > 1e-6 else 0.3
    pitch += math.radians(3)  # basic gravity offset
    pitch = max(math.radians(-10), min(math.radians(75), pitch))
    return pitch, yaw


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
    target_id = data.get("target_id")  # Optional: pre-selected target
    aim_direction = data.get("aim_direction")  # Mouse aim direction (Three.js coords)
    command = data.get("command")  # NL command (RL mode)

    # Resolve a reference target (for auto-aim or miss reporting)
    target_obj = None
    if target_id:
        target_obj = registry.get_by_id(target_id)
    if target_obj is None and command:
        resolved = pipeline.ground(command, registry, AGENT_POS)
        if resolved:
            target_obj = resolved.target_obj
    # Fallback: use closest target
    if target_obj is None:
        all_targets = registry.get_all_active()
        if not all_targets:
            return jsonify({"error": "No targets in scene", "hit": False}), 200
        target_obj = min(all_targets, key=lambda o: np.linalg.norm(o.position - ARROW_ORIGIN))

    # Build wind model from request
    wind_data = data.get("wind", {})
    wind_speed = wind_data.get("speed", 0.0)
    wind_dir = wind_data.get("direction", [0, 0, 0])
    wind = WindModel(
        direction=np.array(wind_dir, dtype=float),
        speed=float(wind_speed),
        gust_variance=float(wind_speed) * 0.1,
    )

    # Compute aim based on mode
    mode = data.get("mode", "heuristic")
    draw_strength = 1.0

    if mode == "rl" and rl_loaded:
        # RL agent decides aim + draw strength
        pitch, yaw, draw_strength = _rl_aim(target_obj, wind)
    elif aim_direction and isinstance(aim_direction, dict):
        pitch, yaw = _aim_from_direction(aim_direction, wind)
    else:
        pitch, yaw = _heuristic_aim(target_obj.position.copy(), wind)

    vel = compute_launch_velocity(pitch, yaw, draw_strength, STANDARD_ARROW)

    # Simulate
    traj = simulate_trajectory(
        ARROW_ORIGIN.copy(), vel, STANDARD_ARROW, wind, dt=0.005, max_time=5.0,
    )

    # Find trajectory landing point (where z <= 0 or last point)
    landing_pos = traj[-1][0] if traj else ARROW_ORIGIN.copy()
    apex_z = max(p[2] for p, _ in traj) if traj else 0
    for pos, _ in traj:
        if pos[2] <= 0:
            landing_pos = pos
            break

    # Check hit against ALL targets
    hit = False
    hit_pos = None
    hit_target = None
    closest_target = None
    closest_dist = float("inf")
    hit_checks = []
    for obj in registry.get_all_active():
        collision_target = Target(
            id=obj.id,
            position=obj.position.copy(),
            radius=obj.radius,
        )
        obj_hit, obj_hit_pos, obj_dist = check_hit(traj, collision_target)
        hit_checks.append({
            "id": obj.id, "color": obj.flag_color,
            "target_pos": obj.position.tolist(),
            "target_pos_threejs": _physics_to_threejs(obj.position),
            "radius": obj.radius, "hit": obj_hit,
            "closest_dist": round(obj_dist, 4) if obj_dist else None,
            "hit_point": obj_hit_pos.tolist() if obj_hit_pos is not None else None,
        })
        if obj_hit:
            hit = True
            hit_pos = obj_hit_pos
            hit_target = obj
            break
        if obj_dist is not None and obj_dist < closest_dist:
            closest_dist = obj_dist
            closest_target = obj

    # Build comprehensive log entry
    log = _make_logger("rl", "rl_shots.log") if mode == "rl" else _make_logger("heuristic", "heuristic_shots.log")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep = "=" * 80
    log.info(sep)
    log.info(f"SHOT @ {ts}  |  mode={mode}")
    log.info(f"  INTENDED TARGET: {target_obj.id} ({target_obj.flag_color})")
    log.info(f"    physics pos:   {target_obj.position.tolist()}")
    log.info(f"    threejs pos:   {_physics_to_threejs(target_obj.position)}")
    log.info(f"    distance:      {np.linalg.norm(target_obj.position - ARROW_ORIGIN):.2f}m")
    log.info(f"  AIM PARAMETERS:")
    log.info(f"    pitch:         {np.degrees(pitch):.2f}°")
    log.info(f"    yaw:           {np.degrees(yaw):.2f}°")
    log.info(f"    draw_strength: {draw_strength:.3f}")
    if mode == "rl":
        obs = _build_rl_observation(target_obj, wind)
        log.info(f"    rl_obs (raw):  {obs.tolist()}")
        if rl_vec_normalize is not None:
            obs_norm = rl_vec_normalize.normalize_obs(obs)
            log.info(f"    rl_obs (norm): {obs_norm.tolist()}")
        action, _ = rl_model.predict(obs if rl_vec_normalize is None else rl_vec_normalize.normalize_obs(obs), deterministic=True)
        log.info(f"    rl_action:     {action.tolist()}")
    elif aim_direction:
        log.info(f"    mouse_aim_dir: {aim_direction}")
    log.info(f"  ARROW:")
    log.info(f"    origin:        {ARROW_ORIGIN.tolist()}")
    log.info(f"    velocity:      [{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}]")
    log.info(f"    speed:         {np.linalg.norm(vel):.2f} m/s")
    log.info(f"    apex_height:   {apex_z:.2f}m")
    log.info(f"    landing_pos:   {landing_pos.tolist()}")
    log.info(f"    traj_points:   {len(traj)}")
    log.info(f"  WIND:")
    log.info(f"    direction:     {wind.direction.tolist()}")
    log.info(f"    speed:         {wind.speed:.2f} m/s")
    log.info(f"  ALL TARGETS (hit check):")
    for hc in hit_checks:
        marker = ">>> HIT <<<" if hc["hit"] else ""
        log.info(f"    {hc['id']:12s} ({hc['color']:6s}) | physics={hc['target_pos']} | threejs={hc['target_pos_threejs']} | r={hc['radius']} | closest={hc['closest_dist']}m {marker}")
        if hc["hit"] and hc["hit_point"]:
            log.info(f"      hit_point: {hc['hit_point']}")
    log.info(f"  RESULT: {'HIT ' + hit_target.id + ' (' + hit_target.flag_color + ')' if hit else 'MISS (nearest: ' + (closest_target.id if closest_target else 'none') + ' @ ' + f'{closest_dist:.2f}m' + ')'}")
    log.info("")

    # If hit, truncate trajectory at the impact point
    if hit and hit_pos is not None:
        truncated = []
        for pos, vel in traj:
            truncated.append((pos, vel))
            if np.linalg.norm(pos - hit_pos) < 0.05:
                break
        # If we didn't find the exact point, find closest
        if len(truncated) == len(traj):
            best_idx = 0
            best_d = float("inf")
            for i, (pos, _) in enumerate(traj):
                d = np.linalg.norm(pos - hit_pos)
                if d < best_d:
                    best_d = d
                    best_idx = i
            truncated = traj[:best_idx + 1]
        # Append exact hit point as final position
        truncated.append((hit_pos.copy(), np.zeros(3)))
        traj_for_frontend = truncated
    else:
        traj_for_frontend = traj

    # Sample trajectory for frontend
    sampled = _sample_trajectory(traj_for_frontend, max_points=120)
    trajectory_points = [_physics_to_threejs(pos) for pos, _ in sampled]

    if hit:
        target_distance = float(np.linalg.norm(hit_target.position - ARROW_ORIGIN))
        result_text = f"Hit {hit_target.flag_color} target ({hit_target.id}) at {target_distance:.1f}m!"
        resp_target = hit_target
    else:
        resp_target = closest_target or target_obj
        target_distance = float(np.linalg.norm(resp_target.position - ARROW_ORIGIN))
        result_text = f"Missed! Nearest was {resp_target.flag_color} target by {closest_dist:.2f}m"

    return jsonify({
        "target_id": resp_target.id,
        "hit": hit,
        "hit_point": _physics_to_threejs(hit_pos) if hit_pos is not None else None,
        "trajectory_points": trajectory_points,
        "target_position": _physics_to_threejs(resp_target.position),
        "result_text": result_text,
        "distance": target_distance,
        "flag_color": resp_target.flag_color,
        "mode": mode,
        "draw_strength": draw_strength,
    })


@app.route("/api/rl_status")
def rl_status():
    """Check if the RL model is loaded and ready."""
    return jsonify({
        "available": rl_loaded,
        "model_loaded": rl_model is not None,
        "vecnorm_loaded": rl_vec_normalize is not None,
    })


@app.route("/api/fire_rl", methods=["POST"])
def fire_rl():
    """Fire at all targets using the RL agent and return results.

    Useful for batch demo — fires one arrow at each target in the scene.
    """
    if not rl_loaded:
        return jsonify({"error": "RL model not loaded"}), 400

    data = request.get_json(force=True)
    target_id = data.get("target_id")

    # Build wind model
    wind_data = data.get("wind", {})
    wind_speed = wind_data.get("speed", 0.0)
    wind_dir = wind_data.get("direction", [0, 0, 0])
    wind = WindModel(
        direction=np.array(wind_dir, dtype=float),
        speed=float(wind_speed),
        gust_variance=float(wind_speed) * 0.1,
    )

    # Get target
    target_obj = None
    if target_id:
        target_obj = registry.get_by_id(target_id)
    if target_obj is None:
        all_targets = registry.get_all_active()
        if not all_targets:
            return jsonify({"error": "No targets in scene"}), 200
        target_obj = min(all_targets, key=lambda o: np.linalg.norm(o.position - ARROW_ORIGIN))

    # RL agent aims
    pitch, yaw, draw_strength = _rl_aim(target_obj, wind)
    vel = compute_launch_velocity(pitch, yaw, draw_strength, STANDARD_ARROW)

    # Simulate
    traj = simulate_trajectory(
        ARROW_ORIGIN.copy(), vel, STANDARD_ARROW, wind, dt=0.005, max_time=5.0,
    )

    # Check hit against ALL targets
    hit = False
    hit_pos = None
    hit_target = None
    closest_target = None
    closest_dist = float("inf")
    for obj in registry.get_all_active():
        collision_target = Target(id=obj.id, position=obj.position.copy(), radius=obj.radius)
        obj_hit, obj_hit_pos, obj_dist = check_hit(traj, collision_target)
        if obj_hit:
            hit = True
            hit_pos = obj_hit_pos
            hit_target = obj
            break
        if obj_dist is not None and obj_dist < closest_dist:
            closest_dist = obj_dist
            closest_target = obj

    # Truncate trajectory at hit point
    if hit and hit_pos is not None:
        truncated = []
        for pos, v in traj:
            truncated.append((pos, v))
            if np.linalg.norm(pos - hit_pos) < 0.05:
                break
        if len(truncated) == len(traj):
            best_idx = min(range(len(traj)), key=lambda i: np.linalg.norm(traj[i][0] - hit_pos))
            truncated = traj[:best_idx + 1]
        truncated.append((hit_pos.copy(), np.zeros(3)))
        traj_for_frontend = truncated
    else:
        traj_for_frontend = traj

    sampled = _sample_trajectory(traj_for_frontend, max_points=120)
    trajectory_points = [_physics_to_threejs(pos) for pos, _ in sampled]

    resp_target = hit_target if hit else (closest_target or target_obj)
    target_distance = float(np.linalg.norm(resp_target.position - ARROW_ORIGIN))

    if hit:
        result_text = f"[RL] Hit {hit_target.flag_color} target at {target_distance:.1f}m!"
    else:
        result_text = f"[RL] Missed! Nearest: {resp_target.flag_color} by {closest_dist:.2f}m"

    return jsonify({
        "target_id": resp_target.id,
        "hit": hit,
        "hit_point": _physics_to_threejs(hit_pos) if hit_pos is not None else None,
        "trajectory_points": trajectory_points,
        "target_position": _physics_to_threejs(resp_target.position),
        "result_text": result_text,
        "distance": target_distance,
        "flag_color": resp_target.flag_color,
        "mode": "rl",
        "draw_strength": draw_strength,
        "aim_pitch_deg": float(np.degrees(pitch)),
        "aim_yaw_deg": float(np.degrees(yaw)),
    })


@app.route("/api/register_bullseyes", methods=["POST"])
def register_bullseyes():
    """Acknowledge GLB bullseye registration but skip adding to registry.
    We use our own procedural targets instead of GLB-extracted ones."""
    data = request.get_json(force=True)
    bullseyes = data.get("bullseyes", [])
    # Don't register — our TARGET_CONFIGS targets are the only valid targets
    return jsonify({"registered": 0, "ids": [], "note": "Using procedural targets"})


@app.route("/api/randomize", methods=["POST"])
def randomize():
    _generate_scene()
    return jsonify(_scene_json())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
