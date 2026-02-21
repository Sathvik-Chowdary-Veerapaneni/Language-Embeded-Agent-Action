"""
LEAA RL Training — Main Training Script

Runs PPO through curriculum stages with Stable-Baselines3.
Default device: CPU (MPS available via --device mps but slower for MlpPolicy).

Usage:
    python rl_training/train.py
    python rl_training/train.py --device cpu --stage 0
    python rl_training/train.py --resume rl_training/checkpoints/static_close_best.zip
"""

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml
import torch
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rl_training.envs.archery_env import ArcheryEnv

console = Console()

# ---------- Paths ----------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"
CHECKPOINTS_DIR = Path(__file__).resolve().parent / "checkpoints"
LOGS_DIR = Path(__file__).resolve().parent / "logs"


def load_curriculum(config_path: Path = None) -> list:
    """Load curriculum stages from YAML."""
    if config_path is None:
        config_path = CONFIGS_DIR / "curriculum.yaml"
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return data["stages"]


def get_device(requested: str = "cpu") -> str:
    """Determine best available device."""
    if requested == "mps":
        if torch.backends.mps.is_available():
            console.print("[yellow]⚠ MPS selected — note: SB3 MlpPolicy is slower on MPS than CPU[/yellow]")
            return "mps"
        else:
            console.print("[yellow]⚠ MPS not available, falling back to CPU[/yellow]")
            return "cpu"
    return "cpu"


# ---------- Custom Callback ----------
class CurriculumCallback(BaseCallback):
    """Tracks success rate and manages curriculum progression."""

    def __init__(
        self,
        stages: list,
        current_stage: int,
        checkpoint_dir: Path,
        window_size: int = 1000,
        checkpoint_freq: int = 50000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.stages = stages
        self.current_stage = current_stage
        self.checkpoint_dir = checkpoint_dir
        self.window_size = window_size
        self.checkpoint_freq = checkpoint_freq

        self.hit_history = deque(maxlen=window_size)
        self.episode_count = 0
        self.best_success_rate = 0.0
        self.stage_episode_count = 0
        self.last_checkpoint_step = 0

    @property
    def current_stage_config(self) -> dict:
        return self.stages[self.current_stage]

    @property
    def success_rate(self) -> float:
        if len(self.hit_history) == 0:
            return 0.0
        return sum(self.hit_history) / len(self.hit_history)

    def _on_step(self) -> bool:
        # Check infos for episode completion
        infos = self.locals.get("infos", [])
        for info in infos:
            if "hit" in info:
                self.hit_history.append(1.0 if info["hit"] else 0.0)
                self.episode_count += 1
                self.stage_episode_count += 1

        # Log metrics
        if self.episode_count > 0 and self.episode_count % 100 == 0:
            self.logger.record("curriculum/stage", self.current_stage)
            self.logger.record("curriculum/stage_name", self.current_stage_config["name"])
            self.logger.record("curriculum/success_rate", self.success_rate)
            self.logger.record("curriculum/episodes", self.episode_count)

        # Checkpoint
        if self.num_timesteps - self.last_checkpoint_step >= self.checkpoint_freq:
            self._save_checkpoint()
            self.last_checkpoint_step = self.num_timesteps

        # Best model
        if self.success_rate > self.best_success_rate and len(self.hit_history) >= 100:
            self.best_success_rate = self.success_rate
            self._save_best()

        # Stage advancement
        stage_cfg = self.current_stage_config
        threshold = stage_cfg.get("success_threshold", 0.9)
        min_eps = stage_cfg.get("min_episodes", 50000)

        if (
            self.success_rate >= threshold
            and self.stage_episode_count >= min_eps
            and len(self.hit_history) >= self.window_size
            and self.current_stage < len(self.stages) - 1
        ):
            self._advance_stage()

        return True

    def _save_checkpoint(self) -> None:
        stage_name = self.current_stage_config["name"]
        path = self.checkpoint_dir / f"{stage_name}_step_{self.num_timesteps}.zip"
        self.model.save(str(path))
        if self.verbose:
            console.print(f"  💾 Checkpoint saved: {path.name} (success: {self.success_rate:.1%})")

    def _save_best(self) -> None:
        stage_name = self.current_stage_config["name"]
        path = self.checkpoint_dir / f"{stage_name}_best.zip"
        self.model.save(str(path))

    def _advance_stage(self) -> None:
        old_name = self.current_stage_config["name"]
        self.current_stage += 1
        new_cfg = self.current_stage_config

        console.print(f"\n[bold green]🎯 Stage advanced: {old_name} → {new_cfg['name']}[/bold green]")
        console.print(f"   Success rate was: {self.success_rate:.1%}")

        # Reset tracking for new stage
        self.hit_history.clear()
        self.best_success_rate = 0.0
        self.stage_episode_count = 0

        # Update environment config
        self.training_env.env_method("_update_config", new_cfg)


# ---------- Training Function ----------
def train(
    device: str = "cpu",
    start_stage: int = 0,
    resume_path: str = None,
    total_timesteps: int = 2_000_000,
    quick_test: bool = False,
):
    """Main training loop with curriculum progression."""

    # Load curriculum
    stages = load_curriculum()
    stage_config = stages[start_stage]

    console.print(f"\n[bold cyan]═══ LEAA Training ═══[/bold cyan]")
    console.print(f"  Device: {device}")
    console.print(f"  Starting stage: {start_stage} ({stage_config['name']})")
    console.print(f"  Total timesteps: {total_timesteps:,}")

    # Dirs
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = ArcheryEnv(stage_config=stage_config)

    # Create or load model
    if resume_path:
        console.print(f"  Resuming from: {resume_path}")
        model = PPO.load(resume_path, env=env, device=device)
    else:
        policy_kwargs = dict(
            net_arch=[256, 256, 128],
        )

        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            batch_size=256,
            n_steps=2048,
            gamma=0.99,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=0,
            device=device,
            tensorboard_log=str(LOGS_DIR),
        )

    # Setup TensorBoard logger
    logger = configure(str(LOGS_DIR / f"ppo_{stage_config['name']}"), ["stdout", "tensorboard"])
    model.set_logger(logger)

    console.print(f"  Policy network: {model.policy}")

    # Quick test mode
    if quick_test:
        total_timesteps = 2048  # Minimum for one PPO update

    # Callback
    callback = CurriculumCallback(
        stages=stages,
        current_stage=start_stage,
        checkpoint_dir=CHECKPOINTS_DIR,
        checkpoint_freq=50000 if not quick_test else 1024,
    )

    # Train
    console.print(f"\n[bold]Starting training...[/bold]\n")
    start_time = time.time()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True,
            tb_log_name=f"ppo_{stage_config['name']}",
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")

    elapsed = time.time() - start_time

    # Final save
    final_path = CHECKPOINTS_DIR / f"final_stage{callback.current_stage}.zip"
    model.save(str(final_path))

    # Summary
    console.print(f"\n[bold cyan]═══ Training Summary ═══[/bold cyan]")
    console.print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    console.print(f"  Episodes: {callback.episode_count:,}")
    console.print(f"  Final stage: {callback.current_stage} ({callback.current_stage_config['name']})")
    console.print(f"  Final success rate: {callback.success_rate:.1%}")
    console.print(f"  Model saved: {final_path}")

    return model


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="LEAA RL Training")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps"],
                        help="Training device (default: cpu)")
    parser.add_argument("--stage", type=int, default=0,
                        help="Starting curriculum stage (default: 0)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--timesteps", type=int, default=2_000_000,
                        help="Total training timesteps")
    parser.add_argument("--quick-test", action="store_true",
                        help="Quick test mode (minimal training)")
    args = parser.parse_args()

    device = get_device(args.device)
    train(
        device=device,
        start_stage=args.stage,
        resume_path=args.resume,
        total_timesteps=args.timesteps,
        quick_test=args.quick_test,
    )


if __name__ == "__main__":
    main()
