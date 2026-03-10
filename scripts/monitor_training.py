"""
LEAA Training Monitor — run locally to track Colab progress
============================================================
Usage:
    python scripts/monitor_training.py          # one-shot status
    python scripts/monitor_training.py --watch  # refresh every 60s
"""

import argparse
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLOUD_DIR = REPO_ROOT / "cloud_checkpoints"

STAGES = [
    {"num": 3, "name": "static_far",   "target": 0.85},
    {"num": 4, "name": "moving_slow",  "target": 0.70},
    {"num": 5, "name": "wind",         "target": 0.55},
    {"num": 6, "name": "full_dynamic", "target": 0.35},
]


def git_pull():
    result = subprocess.run(
        ["git", "pull", "--rebase"],
        cwd=REPO_ROOT, capture_output=True, text=True
    )
    return result.stdout.strip()


def get_checkpoint_info(stage_name: str) -> dict:
    stage_dir = CLOUD_DIR / stage_name
    best_zip = stage_dir / f"{stage_name}_best.zip"
    best_pkl = stage_dir / f"vecnormalize_{stage_name}_best.pkl"

    if not best_zip.exists():
        return {"status": "no checkpoint", "mtime": None, "age_hrs": None}

    mtime = best_zip.stat().st_mtime
    age_hrs = (time.time() - mtime) / 3600
    return {
        "status": "checkpoint found",
        "mtime": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "age_hrs": age_hrs,
        "has_pkl": best_pkl.exists(),
    }


def print_status():
    print(f"\n{'='*58}")
    print(f"  LEAA Training Monitor — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*58}")

    pull_msg = git_pull()
    if "Already up to date" in pull_msg:
        print(f"  Git: up to date")
    else:
        print(f"  Git: {pull_msg}")

    print(f"\n  {'Stage':<15} {'Target':>7}  {'Last Checkpoint':<22} {'Age':>8}  Status")
    print(f"  {'-'*70}")

    for s in STAGES:
        info = get_checkpoint_info(s["name"])
        target_str = f"{s['target']:.0%}"

        if info["mtime"]:
            age = info["age_hrs"]
            if age < 1:
                age_str = f"{age*60:.0f}m ago"
                status = "🟢 ACTIVE" if age < 0.6 else "🟡 idle"
            elif age < 6:
                age_str = f"{age:.1f}h ago"
                status = "🟡 idle"
            else:
                age_str = f"{age:.1f}h ago"
                status = "🔴 stale"
            pkl_str = "✓" if info["has_pkl"] else "✗ no pkl"
            print(f"  {s['name']:<15} {target_str:>7}  {info['mtime']:<22} {age_str:>8}  {status} {pkl_str}")
        else:
            print(f"  {s['name']:<15} {target_str:>7}  {'—':<22} {'—':>8}  ⚪ no checkpoint")

    print(f"\n  Tip: Colab pushes every 30 min — age <30m = actively training")
    print(f"{'='*58}\n")


def main():
    parser = argparse.ArgumentParser(description="LEAA Training Monitor")
    parser.add_argument("--watch", action="store_true",
                        help="Continuously refresh every 60 seconds")
    parser.add_argument("--interval", type=int, default=60,
                        help="Refresh interval in seconds (default: 60)")
    args = parser.parse_args()

    if args.watch:
        print("Watching... (Ctrl+C to stop)")
        while True:
            print_status()
            time.sleep(args.interval)
    else:
        print_status()


if __name__ == "__main__":
    main()
