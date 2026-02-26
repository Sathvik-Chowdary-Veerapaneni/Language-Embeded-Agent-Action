"""
LEAA RL Training — Main Training Script

Runs PPO through curriculum stages with Stable-Baselines3.
Supports vectorized environments (SubprocVecEnv) for parallel rollout collection.
Default device: CPU (MPS available via --device mps but slower for MlpPolicy).

Usage:
    python rl_training/train.py --device cpu --num-envs 8
    python rl_training/train.py --device cpu --num-envs 8 --timesteps 200000
    python rl_training/train.py --resume rl_training/checkpoints/static_close_best.zip --num-envs 8
    # v3.1 dynamic-only resume:
    python rl_training/train.py --resume rl_training/checkpoints/final_stage2.zip --start-stage 3 --num-envs 8
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
from rich.console import Console

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_linear_fn, set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

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


def make_env(rank: int, seed: int, stage_config: dict, log_dir: Path = None):
    """Factory function that returns a callable to create a Monitor-wrapped ArcheryEnv."""
    def _init():
        env = ArcheryEnv(stage_config=stage_config)
        # Wrap in Monitor for proper TensorBoard logging (ep_rew_mean, ep_len_mean)
        monitor_path = str(log_dir / f"monitor_{rank}") if log_dir else None
        env = Monitor(env, filename=monitor_path)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed + rank)
    return _init


def create_vec_env(num_envs: int, stage_config: dict, seed: int = 42, log_dir: Path = None):
    """Create a vectorized environment with Monitor wrappers."""
    if num_envs > 1:
        env = SubprocVecEnv(
            [make_env(i, seed, stage_config, log_dir) for i in range(num_envs)],
            start_method="fork",
        )
    else:
        env = DummyVecEnv([make_env(0, seed, stage_config, log_dir)])
    return env


# ---------- Custom Callbacks ----------

class CurriculumCallback(BaseCallback):
    """Tracks success rate and manages curriculum progression."""

    def __init__(
        self,
        stages: list,
        current_stage: int,
        checkpoint_dir: Path,
        num_envs: int = 1,
        window_size: int = 1000,
        checkpoint_freq: int = 50000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.stages = stages
        self.current_stage = current_stage
        self.checkpoint_dir = checkpoint_dir
        self.num_envs = num_envs
        self.window_size = window_size
        self.checkpoint_freq = checkpoint_freq

        self.hit_history = deque(maxlen=window_size)
        self.episode_count = 0
        self.best_success_rate = 0.0
        self.stage_episode_count = 0
        self.last_checkpoint_step = 0
        self.stage_complete = False  # Signal to outer loop for stage advancement

    @property
    def current_stage_config(self) -> dict:
        return self.stages[self.current_stage]

    @property
    def success_rate(self) -> float:
        if len(self.hit_history) == 0:
            return 0.0
        return sum(self.hit_history) / len(self.hit_history)

    def _on_step(self) -> bool:
        # Check infos for episode completion (vectorized: multiple infos per step)
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
            self.logger.record("curriculum/envs", self.num_envs)

        # Checkpoint
        if self.num_timesteps - self.last_checkpoint_step >= self.checkpoint_freq:
            self._save_checkpoint()
            self.last_checkpoint_step = self.num_timesteps

        # Best model
        if self.success_rate > self.best_success_rate and len(self.hit_history) >= 100:
            self.best_success_rate = self.success_rate
            self._save_best()

        # Stage advancement check — signal to outer loop (don't swap env here)
        stage_cfg = self.current_stage_config
        threshold = stage_cfg.get("success_threshold", 0.9)
        min_eps = stage_cfg.get("min_episodes", 50000)

        if (
            self.success_rate >= threshold
            and self.stage_episode_count >= min_eps
            and len(self.hit_history) >= self.window_size
            and self.current_stage < len(self.stages) - 1
        ):
            self.stage_complete = True
            return False  # Stop current model.learn() to trigger stage swap

        return True

    def _save_checkpoint(self) -> None:
        stage_name = self.current_stage_config["name"]
        path = self.checkpoint_dir / f"{stage_name}_step_{self.num_timesteps}.zip"
        self.model.save(str(path))
        # Save VecNormalize stats alongside model
        vecnorm_path = self.checkpoint_dir / f"vecnormalize_{stage_name}_step_{self.num_timesteps}.pkl"
        self.training_env.save(str(vecnorm_path))
        if self.verbose:
            console.print(f"  💾 Checkpoint saved: {path.name} (success: {self.success_rate:.1%})")

    def _save_best(self) -> None:
        stage_name = self.current_stage_config["name"]
        path = self.checkpoint_dir / f"{stage_name}_best.zip"
        self.model.save(str(path))
        vecnorm_path = self.checkpoint_dir / f"vecnormalize_{stage_name}_best.pkl"
        self.training_env.save(str(vecnorm_path))

    def advance_stage(self) -> None:
        """Advance to next curriculum stage. Called by the outer training loop."""
        old_name = self.current_stage_config["name"]
        self.current_stage += 1
        new_cfg = self.current_stage_config

        console.print(f"\n[bold green]🎯 Stage advanced: {old_name} → {new_cfg['name']}[/bold green]")
        console.print(f"   Success rate was: {self.success_rate:.1%}")

        # Reset tracking for new stage
        self.hit_history.clear()
        self.best_success_rate = 0.0
        self.stage_episode_count = 0
        self.stage_complete = False


class EntropyCoefficientSchedule(BaseCallback):
    """Linearly decay ent_coef over training to shift from exploration to exploitation.

    Bug fix (v3.1): progress is computed relative to THIS run's start timestep, not
    the global counter. On resume, num_timesteps starts at ~6.9M which would make
    progress > 1.0 and jump ent_coef straight to end value. We capture start_timesteps
    on the first call so progress always goes 0→1 within this run.
    """

    def __init__(self, start: float = 0.03, end: float = 0.008,
                 total_timesteps: int = 5_000_000, verbose: int = 0):
        super().__init__(verbose)
        self.start = start
        self.end = end
        self.total_timesteps = total_timesteps
        self.start_timesteps = None  # Captured on first _on_step — offsets global counter

    def _on_step(self) -> bool:
        # Capture baseline on first call so resumed runs start at progress=0
        if self.start_timesteps is None:
            self.start_timesteps = self.num_timesteps

        elapsed = self.num_timesteps - self.start_timesteps
        progress = min(elapsed / self.total_timesteps, 1.0)
        new_ent = self.start - (self.start - self.end) * progress
        self.model.ent_coef = new_ent
        # Log every 100 steps to avoid TensorBoard spam
        if self.num_timesteps % 100 == 0:
            self.logger.record("train/ent_coef_scheduled", new_ent)
        return True


class KLSafetyCallback(BaseCallback):
    """Halves LR if approx_kl exceeds threshold for consecutive_limit rollouts.

    Prevents the KL destabilization seen in v2 (0.047) and v3 late training.
    Once triggered, LR stays halved for the rest of training.

    Bug fix (v3.1): moved from _on_step to _on_rollout_end.
    _on_step fires during rollout collection when name_to_value may be empty
    (logger.dump() clears it at end of collect_rollouts, before PPO.train() writes
    new approx_kl). _on_rollout_end fires after rollout collection and BEFORE dump(),
    so name_to_value still contains approx_kl from the previous PPO.train() call.
    This also counts per-rollout (= per update) rather than per env-step.
    """

    def __init__(self, kl_threshold: float = 0.04, consecutive_limit: int = 3, verbose: int = 0):
        super().__init__(verbose)
        self.kl_threshold = kl_threshold
        self.consecutive_limit = consecutive_limit
        self.consecutive_high = 0
        self.lr_halved = False

    def _on_rollout_end(self) -> None:
        """Called once per rollout cycle, before dump() clears name_to_value."""
        kl = self.model.logger.name_to_value.get("train/approx_kl", None)
        if kl is None:
            return  # First rollout: no previous PPO.train() yet, skip
        if kl > self.kl_threshold:
            self.consecutive_high += 1
            if self.consecutive_high >= self.consecutive_limit and not self.lr_halved:
                for param_group in self.model.policy.optimizer.param_groups:
                    param_group['lr'] *= 0.5
                self.lr_halved = True
                console.print(
                    f"\n[bold yellow]⚠️  KL safety triggered: approx_kl={kl:.4f} "
                    f"for {self.consecutive_limit} consecutive rollouts — LR halved[/bold yellow]"
                )
                self.logger.record("train/kl_safety_triggered", 1)
        else:
            self.consecutive_high = 0

    def _on_step(self) -> bool:
        return True


class TightenVecNormalize(BaseCallback):
    """Tighten VecNormalize clip_obs from wide transition value back to normal.

    On resume or stage transition, clip_obs is widened (15-20) to tolerate
    the observation distribution shift. After warmup_steps the running stats
    have adapted enough to tighten back to clip_obs (default 10.0).

    Re-triggerable: call trigger(current_step) after each stage transition.
    """

    def __init__(self, warmup_steps: int = 20000, clip_obs: float = 10.0, verbose: int = 0):
        super().__init__(verbose)
        self.warmup_steps = warmup_steps
        self.clip_obs = clip_obs
        self._target_step: int = None  # Global timestep at which to tighten
        self._active = False

    def trigger(self, current_step: int) -> None:
        """Schedule tightening warmup_steps after current_step."""
        self._target_step = current_step + self.warmup_steps
        self._active = True

    def _on_step(self) -> bool:
        if self._active and self._target_step is not None and self.num_timesteps >= self._target_step:
            self.model.env.clip_obs = self.clip_obs
            self._active = False
            console.print(f"\n[cyan]📐 VecNormalize clip_obs tightened to {self.clip_obs}[/cyan]")
        return True


# ---------- Training Function ----------
def train(
    device: str = "cpu",
    start_stage: int = 0,
    resume_path: str = None,
    resume_stage: int = None,   # --start-stage: skip early stages, budget → stages resume_stage+
    total_timesteps: int = 5_000_000,
    num_envs: int = 8,
    quick_test: bool = False,
):
    """Main training loop with curriculum progression and vectorized envs."""

    # v3: 15 envs caused KL spikes to 0.047. Cap at 12 with a warning.
    if num_envs > 12:
        console.print(f"  [yellow]⚠ num_envs={num_envs} may cause KL instability "
                      f"(v3 saw spikes at 15). Capping to 12.[/yellow]")
        num_envs = 12

    # --start-stage controls which stage to begin at (independent of --resume).
    # --resume controls checkpoint loading (independent of --start-stage).
    # is_fine_tune=True only when BOTH are given (conservative hyperparams).
    effective_start_stage = resume_stage if resume_stage is not None else start_stage
    is_fine_tune = (resume_path is not None and resume_stage is not None)

    # Load curriculum
    stages = load_curriculum()
    stage_config = stages[effective_start_stage]

    console.print(f"\n[bold cyan]═══ LEAA Training ═══[/bold cyan]")
    console.print(f"  Device: {device}")
    console.print(f"  Parallel envs: {num_envs}")
    console.print(f"  Total timesteps: {total_timesteps:,}")

    if effective_start_stage > 0:
        skipped = [stages[i]["name"] for i in range(effective_start_stage)]
        console.print(
            f"  [bold green]Skipping stages 0-{effective_start_stage - 1} (already trained): "
            f"{skipped}[/bold green]"
        )

    console.print(f"  Starting stage: {effective_start_stage} ({stage_config['name']})")
    console.print(f"  Batch size: n_steps({2048}) × envs({num_envs}) = {2048 * num_envs:,} per update")

    # Dirs
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Create base vectorized environment
    base_vec_env = create_vec_env(num_envs, stage_config, log_dir=LOGS_DIR)

    # ---------- Model setup ----------
    if resume_path:
        console.print(f"  Resuming from: {resume_path}")

        # VecNormalize candidates — for final_stage2.zip → vecnormalize_final_stage2.pkl
        resume_basename = os.path.basename(resume_path).replace(".zip", "")
        resume_dir = os.path.dirname(resume_path)
        vecnorm_candidates = [
            os.path.join(resume_dir, f"vecnormalize_{resume_basename}.pkl"),
        ]
        # Also try the final pkl from the last completed stage
        if resume_stage is not None and resume_stage > 0:
            prev_stage_name = stages[resume_stage - 1]["name"]
            vecnorm_candidates += [
                str(CHECKPOINTS_DIR / f"vecnormalize_{prev_stage_name}_final.pkl"),
                str(CHECKPOINTS_DIR / f"vecnormalize_{prev_stage_name}_best.pkl"),
            ]

        vecnorm_loaded = False
        for vn_path in vecnorm_candidates:
            if os.path.exists(vn_path):
                env = VecNormalize.load(vn_path, base_vec_env)
                env.training = True
                env.clip_obs = 15.0   # Wide tolerance for stage transition obs distribution shift
                console.print(f"  Loaded VecNormalize stats: {os.path.basename(vn_path)}")
                console.print(f"  VecNormalize: training=True, clip_obs=15.0 (widens for resume transition)")
                vecnorm_loaded = True
                break

        if not vecnorm_loaded:
            console.print("  [yellow]⚠ No VecNormalize stats found — using fresh stats, clip_obs=15.0[/yellow]")
            env = VecNormalize(base_vec_env, norm_obs=True, norm_reward=True,
                               clip_obs=15.0, clip_reward=10.0)

        model = PPO.load(resume_path, env=env, device=device)

        # Override with conservative fine-tuning hyperparams when using --start-stage.
        # These prevent destroying the static policy that already works well.
        if is_fine_tune:
            model.learning_rate = lambda _: 1e-4   # Fixed — no schedule, stable fine-tuning
            model.batch_size = 256                  # Smaller = gentler updates
            model.n_epochs = 3                      # Was 5 — prevents clip_fraction spike (v3: 34%)
            model.ent_coef = 0.03                   # Start lower — policy already has structure
            model.clip_range = lambda _: 0.15       # Tighter leash — prevents positive policy_gradient_loss
            # Sync optimizer LR immediately
            for param_group in model.policy.optimizer.param_groups:
                param_group['lr'] = 1e-4
            console.print(f"  [bold]Fine-tuning hyperparams (resume mode):[/bold]")
            console.print(f"    lr=1e-4 (fixed), batch_size=256, n_epochs=3, ent_coef=0.03, clip_range=0.15")

    else:
        env = VecNormalize(base_vec_env, norm_obs=True, norm_reward=True,
                           clip_obs=10.0, clip_reward=10.0)
        console.print("  VecNormalize: obs=True, reward=True, clip=10.0")

        # v2/v3: Separate policy and value networks
        policy_kwargs = dict(
            net_arch=dict(
                pi=[256, 256],     # actor
                vf=[256, 256],     # critic — independent from actor
            ),
            activation_fn=torch.nn.Tanh,
        )

        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=get_linear_fn(3e-4, 1e-4, 1.0),  # 3e-4 → 1e-4 over training
            n_steps=2048,
            batch_size=512,
            n_epochs=5,                  # v2: reduced from 10 for larger batch
            gamma=1.0,                   # Single-step episodes
            clip_range=0.2,
            ent_coef=0.05,               # Initial — callback will schedule decay
            max_grad_norm=0.5,
            verbose=1,
            device=device,
            tensorboard_log=str(LOGS_DIR),
        )

    # Setup TensorBoard logger
    logger = configure(str(LOGS_DIR / f"ppo_{stage_config['name']}"), ["stdout", "tensorboard"])
    model.set_logger(logger)

    console.print(f"  Policy network: {model.policy}")

    # Quick test mode
    if quick_test:
        total_timesteps = 2048 * num_envs

    # ---------- Per-stage timestep budget ----------
    # Budget always allocated over stages[effective_start_stage:] only.
    # Named weights cover the dynamic stages; earlier stages use fallback weight 1.
    named_stage_weights = {
        "static_close": 1,
        "static_medium": 1,
        "static_mid_far": 2,
        "static_far": 2,
        "moving_slow": 3,
        "wind": 4,
        "full_dynamic": 5,
    }
    remaining_stages = stages[effective_start_stage:]
    remaining_weights = [named_stage_weights.get(s["name"], 1) for s in remaining_stages]
    total_weight = sum(remaining_weights)
    timesteps_per_stage = [
        int(total_timesteps * w / total_weight) for w in remaining_weights
    ]
    console.print(f"\n  [bold]Stage budget allocation "
                  f"(stages {effective_start_stage}+, total={total_timesteps:,}):[/bold]")
    for s, ts, w in zip(remaining_stages, timesteps_per_stage, remaining_weights):
        console.print(f"    stage {stages.index(s)}: {s['name']:15s}: {ts:>10,} steps  (weight={w})")

    # Track which budget index we're on (advances with each stage transition)
    stage_budget_idx = 0

    # ---------- Callbacks ----------
    curriculum_cb = CurriculumCallback(
        stages=stages,
        current_stage=effective_start_stage,
        checkpoint_dir=CHECKPOINTS_DIR,
        num_envs=num_envs,
        checkpoint_freq=50000 if not quick_test else 1024,
    )

    # Entropy schedule: conservative for fine-tune (policy already has structure)
    entropy_cb = EntropyCoefficientSchedule(
        start=0.03 if is_fine_tune else 0.05,
        end=0.008 if is_fine_tune else 0.01,
        total_timesteps=total_timesteps,
    )

    # KL safety: halve LR if approx_kl spikes (prevents v2/v3 destabilization)
    kl_safety_cb = KLSafetyCallback(kl_threshold=0.04, consecutive_limit=3)

    # VecNormalize tightening: always active (triggered on resume AND stage transitions)
    tighten_cb = TightenVecNormalize(warmup_steps=20000, clip_obs=10.0)

    callbacks = [curriculum_cb, entropy_cb, kl_safety_cb, tighten_cb]

    # Trigger tighten immediately if resuming (clip_obs was widened to 15.0)
    if is_fine_tune:
        tighten_cb.trigger(current_step=0)
        console.print(f"  TightenVecNormalize: clip_obs=15.0 → 10.0 after 20k steps")

    console.print(f"  Active callbacks: CurriculumCallback, EntropyCoefficientSchedule, "
                  f"KLSafetyCallback, TightenVecNormalize")

    callback = CallbackList(callbacks)

    # ---------- Training loop ----------
    console.print(f"\n[bold]Starting training...[/bold]")
    console.print(f"  Stage budgets: {[f'{t:,}' for t in timesteps_per_stage]}\n")
    start_time = time.time()
    timesteps_used = 0

    try:
        while timesteps_used < total_timesteps:
            remaining = total_timesteps - timesteps_used
            current_budget = timesteps_per_stage[min(stage_budget_idx, len(timesteps_per_stage) - 1)]
            stage_budget = min(current_budget, remaining)
            stage_name = curriculum_cb.current_stage_config["name"]

            console.print(f"\n[bold cyan]── Stage {curriculum_cb.current_stage}: {stage_name} "
                          f"(budget: {stage_budget:,} steps, remaining: {remaining:,}) ──[/bold cyan]")

            # Update logger for this stage
            stage_logger = configure(
                str(LOGS_DIR / f"ppo_{stage_name}"), ["stdout", "tensorboard"]
            )
            model.set_logger(stage_logger)

            # Bug 2 fix: re-assert fixed LR before every learn() call in fine-tune mode.
            # PPO.load() restores the old v3 schedule; even after setting model.learning_rate,
            # the optimizer param groups retain the old value until explicitly synced.
            if is_fine_tune:
                for param_group in model.policy.optimizer.param_groups:
                    param_group['lr'] = 1e-4

            model.learn(
                total_timesteps=stage_budget,
                callback=callback,
                progress_bar=True,
                tb_log_name=f"ppo_{stage_name}",
                reset_num_timesteps=False,  # Keep global timestep counter
            )

            timesteps_used += stage_budget

            # Check if stage advancement was triggered
            if curriculum_cb.stage_complete:
                # Save VecNormalize stats before transition
                old_vecnorm_path = CHECKPOINTS_DIR / f"vecnormalize_{stage_name}_final.pkl"
                env.save(str(old_vecnorm_path))

                # Advance callback state
                curriculum_cb.advance_stage()
                stage_budget_idx += 1
                new_cfg = curriculum_cb.current_stage_config

                # Close old env and create fresh env for new stage
                env.close()
                new_vec_env = create_vec_env(num_envs, new_cfg, log_dir=LOGS_DIR)

                # Load VecNormalize stats from previous stage (carry over, don't reset)
                env = VecNormalize.load(str(old_vecnorm_path), new_vec_env)
                env.clip_obs = 20.0   # Wider clip during transition
                env.training = True   # Keep adapting stats

                # Re-assign fresh env to model — also resets the rollout buffer
                model.set_env(env)

                # Schedule clip_obs tightening: 20.0 → 10.0 after 20k warmup steps
                tighten_cb.trigger(current_step=model.num_timesteps)
                console.print(f"   VecNormalize stats carried over, clip_obs=20.0 → 10.0 after 20k steps")
                console.print(f"   Rollout buffer reset for clean transition")
            # If stage not complete, loop continues with another budget chunk on same stage

    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted by user[/yellow]")

    elapsed = time.time() - start_time

    # Final save
    final_path = CHECKPOINTS_DIR / f"final_stage{curriculum_cb.current_stage}.zip"
    model.save(str(final_path))
    vecnorm_final = CHECKPOINTS_DIR / f"vecnormalize_final_stage{curriculum_cb.current_stage}.pkl"
    env.save(str(vecnorm_final))

    # Cleanup
    env.close()

    # Summary
    console.print(f"\n[bold cyan]═══ Training Summary ═══[/bold cyan]")
    console.print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    console.print(f"  Episodes: {curriculum_cb.episode_count:,}")
    console.print(f"  Timesteps used: {timesteps_used:,} / {total_timesteps:,}")
    console.print(f"  Final stage: {curriculum_cb.current_stage} ({curriculum_cb.current_stage_config['name']})")
    console.print(f"  Final success rate: {curriculum_cb.success_rate:.1%}")
    console.print(f"  Model saved: {final_path}")

    return model


# ---------- CLI ----------
def main():
    # Print raw sys.argv immediately — before argparse — to diagnose any shell/quoting issues
    print(f"\n[DEBUG] sys.argv = {sys.argv}\n")

    parser = argparse.ArgumentParser(description="LEAA RL Training")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps"],
                        dest="device",
                        help="Training device (default: cpu)")
    parser.add_argument("--stage", type=int, default=0,
                        dest="stage",
                        help="Starting curriculum stage (default: 0)")
    parser.add_argument("--start-stage", type=int, default=None,
                        dest="start_stage",
                        help="Skip stages before this index; allocate ALL budget to stages start-stage+. "
                             "Example: --start-stage 3")
    parser.add_argument("--resume", type=str, default=None,
                        dest="resume",
                        help="Path to checkpoint .zip to resume from")
    parser.add_argument("--timesteps", type=int, default=5_000_000,
                        dest="timesteps",
                        help="Total training timesteps (default: 5M)")
    parser.add_argument("--num-envs", type=int, default=8,
                        dest="num_envs",
                        help="Number of parallel environments (default: 8)")
    parser.add_argument("--quick-test", action="store_true",
                        dest="quick_test",
                        help="Quick test mode (minimal training)")
    args = parser.parse_args()

    # Print parsed values immediately — before anything else
    console.print(f"\n[bold]Parsed args:[/bold]")
    console.print(f"  --resume       = {args.resume}")
    console.print(f"  --start-stage  = {args.start_stage}")
    console.print(f"  --stage        = {args.stage}")
    console.print(f"  --device       = {args.device}")
    console.print(f"  --num-envs     = {args.num_envs}")
    console.print(f"  --timesteps    = {args.timesteps:,}")
    effective = args.start_stage if args.start_stage is not None else args.stage
    console.print(f"  effective start stage → {effective}")

    device = get_device(args.device)
    train(
        device=device,
        start_stage=args.stage,
        resume_path=args.resume,
        resume_stage=args.start_stage,
        total_timesteps=args.timesteps,
        num_envs=args.num_envs,
        quick_test=args.quick_test,
    )


if __name__ == "__main__":
    main()
