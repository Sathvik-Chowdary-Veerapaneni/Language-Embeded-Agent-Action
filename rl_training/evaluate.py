"""
LEAA RL Training — Evaluation Script

Evaluate a trained model across curriculum stages with VecNormalize stats,
performance metrics, and 3D trajectory visualization.

Usage:
    python rl_training/evaluate.py --model rl_training/checkpoints/final_stage3.zip
    python rl_training/evaluate.py --model rl_training/checkpoints/final_stage3.zip --episodes 200 --visualize
    python rl_training/evaluate.py --random --episodes 50  # Random baseline
    python rl_training/evaluate.py --model rl_training/checkpoints/final_stage3.zip --vecnorm rl_training/checkpoints/vecnormalize_final_stage3.pkl
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
from rich.console import Console
from rich.table import Table

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_training.envs.archery_env import ArcheryEnv

console = Console()

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
CHECKPOINTS_DIR = Path(__file__).resolve().parent / "checkpoints"
LOGS_DIR = Path(__file__).resolve().parent / "logs"
PLOTS_DIR = LOGS_DIR / "eval_plots"


def load_curriculum() -> list:
    """Load curriculum stages."""
    with open(CONFIGS_DIR / "curriculum.yaml") as f:
        return yaml.safe_load(f)["stages"]


def find_vecnorm_stats(model_path: str, vecnorm_path: str = None) -> str:
    """Auto-detect the best VecNormalize stats file for a given model.

    Search order:
        1. Explicit --vecnorm path
        2. Same basename: vecnormalize_{model_basename}.pkl
        3. Final stage stats: vecnormalize_final_stage*.pkl
        4. Best checkpoint for each known stage
    """
    if vecnorm_path and os.path.exists(vecnorm_path):
        return vecnorm_path

    model_dir = Path(model_path).parent
    model_stem = Path(model_path).stem  # e.g. "final_stage3"

    candidates = [
        # Match model name directly
        model_dir / f"vecnormalize_{model_stem}.pkl",
        # Common final patterns
        CHECKPOINTS_DIR / f"vecnormalize_{model_stem}.pkl",
        CHECKPOINTS_DIR / "vecnormalize_final_stage3.pkl",
        CHECKPOINTS_DIR / "vecnormalize_final_stage2.pkl",
        CHECKPOINTS_DIR / "vecnormalize_final_stage1.pkl",
        CHECKPOINTS_DIR / "vecnormalize_final_stage0.pkl",
        # Best checkpoints per stage
        CHECKPOINTS_DIR / "vecnormalize_static_far_best.pkl",
        CHECKPOINTS_DIR / "vecnormalize_static_medium_best.pkl",
        CHECKPOINTS_DIR / "vecnormalize_static_close_best.pkl",
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    return None


def evaluate_stage(
    model,
    stage_config: dict,
    n_episodes: int = 100,
    use_random: bool = False,
    vec_normalize: VecNormalize = None,
) -> dict:
    """Evaluate model on a specific curriculum stage.

    If vec_normalize is provided, observations are normalized through it
    before being passed to the model — matching training conditions.

    Returns dict with: hit_rate, avg_reward, avg_dist_from_center, trajectories.
    """
    env = ArcheryEnv(stage_config=stage_config)

    hits = 0
    total_reward = 0.0
    distances = []
    trajectories = []
    rewards_list = []

    # Stats tracking
    bullseyes = 0
    inner_hits = 0
    outer_hits = 0

    for i in range(n_episodes):
        obs, info = env.reset(seed=i)

        if use_random:
            action = env.action_space.sample()
        else:
            # Normalize observation if VecNormalize is available
            if vec_normalize is not None:
                obs_normalized = vec_normalize.normalize_obs(obs)
            else:
                obs_normalized = obs
            action, _ = model.predict(obs_normalized, deterministic=True)

        obs2, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        rewards_list.append(reward)

        if info["hit"]:
            hits += 1
            if info["distance_from_center"] is not None:
                d = info["distance_from_center"]
                distances.append(d)
                # Classify hit zone
                r = env.target.radius
                if d <= r * 0.1:
                    bullseyes += 1
                elif d <= r * 0.5:
                    inner_hits += 1
                else:
                    outer_hits += 1

        # Save a few trajectories for visualization
        if i < 10 and env.last_trajectory:
            trajectories.append({
                "points": [(p.copy(), v.copy()) for p, v in env.last_trajectory],
                "target_pos": env.target.position.copy(),
                "target_radius": env.target.radius,
                "hit": info["hit"],
                "wind": env.wind.get_wind_vector().copy(),
            })

    env.close()

    return {
        "hit_rate": hits / n_episodes,
        "avg_reward": total_reward / n_episodes,
        "avg_dist_from_center": np.mean(distances) if distances else float("nan"),
        "min_dist_from_center": np.min(distances) if distances else float("nan"),
        "max_dist_from_center": np.max(distances) if distances else float("nan"),
        "hits": hits,
        "total": n_episodes,
        "bullseyes": bullseyes,
        "inner_hits": inner_hits,
        "outer_hits": outer_hits,
        "reward_std": np.std(rewards_list),
        "trajectories": trajectories,
    }


def visualize_trajectories(
    results: dict,
    stage_name: str,
    save_dir: Path,
) -> str:
    """Create 3D trajectory plot and save to file. Returns path."""
    trajectories = results["trajectories"]
    if not trajectories:
        return ""

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    for i, traj_data in enumerate(trajectories):
        points = [p for p, _ in traj_data["points"]]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]

        color = "green" if traj_data["hit"] else "red"
        alpha = 0.8 if traj_data["hit"] else 0.4
        ax.plot(xs, ys, zs, color=color, alpha=alpha, linewidth=1.5, label=f"Shot {i+1}")

        # Start point
        ax.scatter(xs[0], ys[0], zs[0], color="blue", s=30, zorder=5)

        # Target
        tp = traj_data["target_pos"]
        r = traj_data["target_radius"]

        # Draw target as a circle
        theta = np.linspace(0, 2 * np.pi, 50)
        cx = tp[0] + r * np.cos(theta) * 0
        cy = tp[1] + r * np.cos(theta)
        cz = tp[2] + r * np.sin(theta)
        ax.plot(cx, cy, cz, color="orange", linewidth=2, alpha=0.6)

        # Target center
        ax.scatter(tp[0], tp[1], tp[2], color="orange", s=100, marker="*", zorder=5)

        # Wind vector at launch
        wind = traj_data["wind"]
        wind_scale = 3.0
        ax.quiver(
            xs[0], ys[0], zs[0] + 2,
            wind[0] * wind_scale, wind[1] * wind_scale, wind[2] * wind_scale,
            color="cyan", alpha=0.5, arrow_length_ratio=0.2,
        )

    ax.set_xlabel("X (forward) [m]")
    ax.set_ylabel("Y (lateral) [m]")
    ax.set_zlabel("Z (up) [m]")
    ax.set_title(
        f"LEAA Trajectories — {stage_name}\n"
        f"Hit rate: {results['hit_rate']:.1%} | Avg reward: {results['avg_reward']:.1f}",
        fontsize=12,
    )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="green", linewidth=2, label="Hit"),
        Line2D([0], [0], color="red", linewidth=2, label="Miss"),
        Line2D([0], [0], color="cyan", linewidth=2, label="Wind"),
        Line2D([0], [0], marker="*", color="orange", linewidth=0, markersize=12, label="Target"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    # Save
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"trajectories_{stage_name}.png"
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()

    return str(path)


def evaluate(
    model_path: str = None,
    vecnorm_path: str = None,
    n_episodes: int = 100,
    stage_name: str = "all",
    visualize: bool = False,
    use_random: bool = False,
):
    """Main evaluation function with VecNormalize support."""

    console.print(f"\n[bold cyan]═══ LEAA Evaluation ═══[/bold cyan]")

    # Load model
    model = None
    vec_normalize = None

    if not use_random:
        if model_path is None:
            console.print("[red]Error: --model required (or use --random)[/red]")
            return
        console.print(f"  Model: {model_path}")
        model = PPO.load(model_path)

        # Load VecNormalize stats — critical for correct evaluation
        vn_path = find_vecnorm_stats(model_path, vecnorm_path)
        if vn_path:
            # Create a dummy vec env just to load VecNormalize stats
            dummy_env = DummyVecEnv([lambda: Monitor(ArcheryEnv())])
            vec_normalize = VecNormalize.load(vn_path, dummy_env)
            vec_normalize.training = False  # Don't update stats during eval
            vec_normalize.norm_reward = False  # Don't normalize rewards
            console.print(f"  VecNormalize: [green]Loaded from {os.path.basename(vn_path)}[/green]")
            console.print(f"    obs_rms mean range: [{vec_normalize.obs_rms.mean.min():.2f}, {vec_normalize.obs_rms.mean.max():.2f}]")
            console.print(f"    obs_rms var range:  [{vec_normalize.obs_rms.var.min():.4f}, {vec_normalize.obs_rms.var.max():.4f}]")
        else:
            console.print("  VecNormalize: [yellow]⚠ No stats found — using raw observations (results may be inaccurate)[/yellow]")
    else:
        console.print("  Using random policy (baseline)")

    # Load stages
    stages = load_curriculum()

    # Filter stages
    if stage_name != "all":
        stages = [s for s in stages if s["name"] == stage_name]
        if not stages:
            console.print(f"[red]Stage '{stage_name}' not found[/red]")
            return

    console.print(f"  Episodes per stage: {n_episodes}")
    console.print(f"  Stages: {[s['name'] for s in stages]}\n")

    # Results table
    table = Table(title="Evaluation Results")
    table.add_column("Stage", style="cyan")
    table.add_column("Hit Rate", justify="right")
    table.add_column("Avg Reward", justify="right")
    table.add_column("Bullseye", justify="right", style="green")
    table.add_column("Inner", justify="right", style="yellow")
    table.add_column("Outer", justify="right", style="red")
    table.add_column("Avg Dist", justify="right")
    table.add_column("Hits / Total", justify="right")

    all_results = {}

    for stage in stages:
        console.print(f"  Evaluating: {stage['name']}...")
        results = evaluate_stage(
            model=model,
            stage_config=stage,
            n_episodes=n_episodes,
            use_random=use_random,
            vec_normalize=vec_normalize,
        )
        all_results[stage["name"]] = results

        table.add_row(
            stage["name"],
            f"{results['hit_rate']:.1%}",
            f"{results['avg_reward']:.1f}",
            f"{results['bullseyes']}",
            f"{results['inner_hits']}",
            f"{results['outer_hits']}",
            f"{results['avg_dist_from_center']:.3f}" if not np.isnan(results["avg_dist_from_center"]) else "N/A",
            f"{results['hits']}/{results['total']}",
        )

        # Visualize
        if visualize:
            plot_path = visualize_trajectories(results, stage["name"], PLOTS_DIR)
            if plot_path:
                console.print(f"    📊 Plot saved: {plot_path}")

    console.print()
    console.print(table)
    console.print()

    # Overall summary
    total_hits = sum(r["hits"] for r in all_results.values())
    total_eps = sum(r["total"] for r in all_results.values())
    total_bullseyes = sum(r["bullseyes"] for r in all_results.values())
    console.print(f"  Overall: {total_hits}/{total_eps} hits ({total_hits/total_eps:.1%})")
    console.print(f"  Bullseyes: {total_bullseyes}")

    # Cleanup dummy env
    if vec_normalize is not None:
        vec_normalize.close()

    return all_results


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="LEAA RL Evaluation")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained model checkpoint")
    parser.add_argument("--vecnorm", type=str, default=None,
                        help="Path to VecNormalize stats (.pkl). Auto-detected if not provided")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes per stage")
    parser.add_argument("--stage", type=str, default="all",
                        help="Specific stage name or 'all'")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate 3D trajectory plots")
    parser.add_argument("--random", action="store_true",
                        help="Evaluate random policy (baseline)")
    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        n_episodes=args.episodes,
        stage_name=args.stage,
        visualize=args.visualize,
        use_random=args.random,
    )


if __name__ == "__main__":
    main()
