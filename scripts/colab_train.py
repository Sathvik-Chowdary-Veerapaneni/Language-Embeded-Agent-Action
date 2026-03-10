"""
LEAA Colab Training Session Script
===================================
Handles one training stage in a Colab session:
  - Restores checkpoint from cloud_checkpoints/<stage>/
  - Runs training (locks to that stage via --max-stage)
  - Syncs best checkpoint to cloud_checkpoints/<stage>/ every 30 min
  - Pushes to GitHub on each sync and at the end

Usage (in Colab):
    python scripts/colab_train.py --stage 3 --timesteps 15000000 --num-envs 4

Stage map:
    3 = static_far
    4 = moving_slow
    5 = wind
    6 = full_dynamic
"""

import argparse
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "rl_training" / "checkpoints"
CLOUD_DIR = REPO_ROOT / "cloud_checkpoints"

STAGE_NAMES = {
    3: "static_far",
    4: "moving_slow",
    5: "wind",
    6: "full_dynamic",
}


def restore_checkpoint(stage_name: str):
    """Copy cloud_checkpoints/<stage>/ → rl_training/checkpoints/"""
    cloud_stage_dir = CLOUD_DIR / stage_name
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    restored = []
    if cloud_stage_dir.exists():
        for f in cloud_stage_dir.iterdir():
            if f.suffix in (".zip", ".pkl"):
                dest = CHECKPOINT_DIR / f.name
                shutil.copy2(f, dest)
                restored.append(f.name)
    if restored:
        print(f"✓ Restored {len(restored)} checkpoint files for {stage_name}")
    else:
        print(f"⚠ No cloud checkpoint found for {stage_name} — starting fresh")


def sync_checkpoint(stage_name: str):
    """Copy best checkpoint → cloud_checkpoints/<stage>/ and push to GitHub."""
    cloud_stage_dir = CLOUD_DIR / stage_name
    cloud_stage_dir.mkdir(parents=True, exist_ok=True)

    synced = []
    for pattern in [f"{stage_name}_best.zip", f"vecnormalize_{stage_name}_best.pkl"]:
        src = CHECKPOINT_DIR / pattern
        if src.exists():
            dest = cloud_stage_dir / pattern
            if not dest.exists() or src.stat().st_mtime > dest.stat().st_mtime:
                shutil.copy2(src, dest)
                synced.append(pattern)

    if not synced:
        print(f"  No new checkpoint to sync for {stage_name}")
        return

    print(f"✓ Synced: {synced}")

    # Git commit + push
    subprocess.run(["git", "add", "cloud_checkpoints/"], cwd=REPO_ROOT, check=False)
    diff = subprocess.run(
        ["git", "diff", "--staged", "--quiet"], cwd=REPO_ROOT
    )
    if diff.returncode != 0:
        subprocess.run(
            ["git", "commit", "-m", f"checkpoint: {stage_name} best update"],
            cwd=REPO_ROOT, check=False,
        )
        result = subprocess.run(["git", "push"], cwd=REPO_ROOT, check=False)
        if result.returncode == 0:
            print(f"✓ Pushed {stage_name} checkpoint to GitHub")
        else:
            print(f"⚠ Git push failed — will retry next sync")


def background_sync(stage_name: str, interval: int = 1800):
    """Sync checkpoint every interval seconds in a background thread."""
    while True:
        time.sleep(interval)
        try:
            print(f"\n[sync] {stage_name} @ {time.strftime('%H:%M:%S')}")
            sync_checkpoint(stage_name)
        except Exception as e:
            print(f"[sync] Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="LEAA Colab Training Session")
    parser.add_argument("--stage", type=int, required=True, choices=[3, 4, 5, 6],
                        help="Stage index to train (3=static_far, 4=moving_slow, 5=wind, 6=full_dynamic)")
    parser.add_argument("--timesteps", type=int, default=15_000_000,
                        help="Total timesteps for this session (default: 15M)")
    parser.add_argument("--num-envs", type=int, default=4,
                        help="Parallel environments (default: 4 for Colab 2-vCPU)")
    parser.add_argument("--sync-interval", type=int, default=1800,
                        help="Checkpoint sync interval in seconds (default: 1800 = 30min)")
    args = parser.parse_args()

    stage_name = STAGE_NAMES[args.stage]
    print(f"\n{'='*50}")
    print(f"  LEAA Colab Training: {stage_name} (stage {args.stage})")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"  Envs: {args.num_envs}")
    print(f"  Sync interval: {args.sync_interval}s")
    print(f"{'='*50}\n")

    # Restore checkpoint from cloud
    restore_checkpoint(stage_name)

    # Resume path
    resume_path = CHECKPOINT_DIR / f"{stage_name}_best.zip"
    if not resume_path.exists():
        print(f"⚠ No resume checkpoint found at {resume_path.name}")
        resume_path = None

    # Start background sync thread
    sync_thread = threading.Thread(
        target=background_sync,
        args=(stage_name, args.sync_interval),
        daemon=True,
    )
    sync_thread.start()
    print(f"✓ Background sync started (every {args.sync_interval//60} min)\n")

    # Build training command
    cmd = [
        "python", "rl_training/train.py",
        "--device", "cuda",
        "--num-envs", str(args.num_envs),
        "--timesteps", str(args.timesteps),
        "--start-stage", str(args.stage),
        "--max-stage", str(args.stage),   # Lock to this stage only
    ]
    if resume_path:
        cmd += ["--resume", str(resume_path)]
        print(f"▶ Resuming from: {resume_path.name}")
    else:
        print(f"▶ Starting fresh for stage {args.stage}")

    print(f"▶ Command: {' '.join(cmd)}\n")

    # Run training
    subprocess.run(cmd, cwd=REPO_ROOT)

    # Final sync after training ends
    print(f"\n{'='*50}")
    print(f"  Training complete — final sync...")
    print(f"{'='*50}")
    sync_checkpoint(stage_name)
    print("\n✓ Done. Session complete.")


if __name__ == "__main__":
    main()
