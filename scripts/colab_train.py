"""
LEAA Training Session Script
=============================
Runs training with aggressive checkpoint syncing to survive VM disconnects.

Everything is backed up to git every sync cycle:
  - All checkpoints (.zip) and VecNormalize stats (.pkl)
  - TensorBoard logs
  - Training logs

Usage (VM — set env vars once, then just run):
    export LEAA_GMAIL="you@gmail.com"
    export LEAA_GMAIL_PASSWORD="your-app-password"

    python scripts/colab_train.py --stage 4 --timesteps 40000000 --num-envs 50 \
        --auto-advance --success-threshold 0.9 --device cpu --sync-interval 300

Usage (Colab):
    python scripts/colab_train.py --stage 4 --timesteps 15000000 --num-envs 4 \
        --max-runtime-hours 11

Stage map:
    0 = static_close    3 = static_far      5 = wind
    1 = static_medium   4 = moving_slow     6 = full_dynamic
    2 = static_mid_far
"""

import argparse
import json
import os
import shutil
import signal
import smtplib
import subprocess
import sys
import threading
import time
import zipfile
from email.mime.text import MIMEText
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "rl_training" / "checkpoints"
CLOUD_DIR = REPO_ROOT / "cloud_checkpoints"
LOGS_DIR = REPO_ROOT / "rl_training" / "logs"

STAGE_NAMES = {
    0: "static_close",
    1: "static_medium",
    2: "static_mid_far",
    3: "static_far",
    4: "moving_slow",
    5: "wind",
    6: "full_dynamic",
}


# ─── Email ────────────────────────────────────────────────────────────────────

class EmailNotifier:
    """Send Gmail notifications via SMTP App Password.

    Credentials are resolved in order:
      1. Explicit arguments (--gmail / --gmail-password)
      2. Environment variables (LEAA_GMAIL / LEAA_GMAIL_PASSWORD)

    To set up once on the VM:
        export LEAA_GMAIL="you@gmail.com"
        export LEAA_GMAIL_PASSWORD="your-app-password"
    """

    def __init__(self, gmail_address: str = None, app_password: str = None):
        self.address = gmail_address or os.environ.get("LEAA_GMAIL")
        self.password = app_password or os.environ.get("LEAA_GMAIL_PASSWORD")
        self.enabled = bool(self.address and self.password)
        if self.enabled:
            source = "args" if gmail_address else "env"
            print(f"✓ Email notifications enabled → {self.address} (from {source})")
        else:
            print("⚠ Email notifications disabled (set LEAA_GMAIL + LEAA_GMAIL_PASSWORD env vars, or pass --gmail)")

    def send(self, subject: str, body: str) -> bool:
        if not self.enabled:
            return False
        try:
            msg = MIMEText(body)
            msg["Subject"] = f"[LEAA] {subject}"
            msg["From"] = self.address
            msg["To"] = self.address
            with smtplib.SMTP("smtp.gmail.com", 587, timeout=15) as smtp:
                smtp.starttls()
                smtp.login(self.address, self.password)
                smtp.sendmail(self.address, self.address, msg.as_string())
            print(f"  📧 Email sent: {subject}")
            return True
        except Exception as e:
            print(f"  ⚠ Email failed: {e}")
            return False


# ─── Runtime Watchdog ─────────────────────────────────────────────────────────

class RuntimeWatchdog:
    """
    Monitors elapsed session time.
    - Sends a warning email at (limit - 1h - 5min)
    - Sends SIGINT to the training process at (limit - 1h) for graceful stop
    """

    def __init__(self, max_runtime_hours: float, notifier: EmailNotifier, stage_name: str):
        self.max_seconds = max_runtime_hours * 3600
        self.buffer_seconds = 3600
        self.warning_lead = 300
        self.notifier = notifier
        self.stage_name = stage_name
        self.start_time = time.time()
        self._proc: subprocess.Popen = None
        self._warned = False
        self._stopped = False

    def register_process(self, proc: subprocess.Popen):
        self._proc = proc

    def start(self):
        t = threading.Thread(target=self._watch, daemon=True)
        t.start()

    def stop(self):
        self._stopped = True

    @staticmethod
    def _fmt(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"

    def _watch(self):
        while not self._stopped:
            time.sleep(30)
            elapsed = time.time() - self.start_time
            remaining = self.max_seconds - elapsed

            if not self._warned and remaining <= (self.buffer_seconds + self.warning_lead):
                self._warned = True
                print(f"\n⚠ Runtime warning: ~1h until graceful stop "
                      f"(elapsed {self._fmt(elapsed)}, remaining {self._fmt(remaining)})")
                # Include live stats in warning
                status = read_training_status() if 'read_training_status' in dir() else {}
                stats_str = ""
                if status:
                    stats_str = (
                        f"\n\nCurrent stats:\n"
                        f"  Stage: {status.get('stage_name', '?')}\n"
                        f"  Success rate: {status.get('success_rate', 0):.1%}\n"
                        f"  Timesteps: {status.get('timesteps', 0):,}\n"
                    )
                self.notifier.send(
                    f"⚠ 1 HOUR LEFT — {self.stage_name}",
                    f"Session approaching time limit.\n"
                    f"Elapsed: {self._fmt(elapsed)}, Remaining: ~{self._fmt(remaining)}\n"
                    f"Training will stop gracefully in ~1 hour.{stats_str}"
                )

            if not self._stopped and remaining <= self.buffer_seconds:
                self._stopped = True
                print(f"\n🛑 Runtime watchdog: stopping training "
                      f"(elapsed {self._fmt(elapsed)}, 1h buffer remaining)")
                if self._proc and self._proc.poll() is None:
                    try:
                        self._proc.send_signal(signal.SIGINT)
                        print("  ✓ SIGINT sent — training process saving checkpoint...")
                    except Exception as e:
                        print(f"  ⚠ Failed to send SIGINT: {e}")
                break


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def is_valid_zip(path: Path) -> bool:
    """Return True if the zip file is not corrupted."""
    try:
        with zipfile.ZipFile(path, "r") as z:
            return z.testzip() is None
    except Exception:
        return False


def restore_checkpoint(stage_name: str, notifier: EmailNotifier):
    """Copy cloud_checkpoints/<stage>/ → rl_training/checkpoints/ with integrity check."""
    cloud_stage_dir = CLOUD_DIR / stage_name
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    restored, skipped = [], []

    if cloud_stage_dir.exists():
        for f in cloud_stage_dir.iterdir():
            if f.suffix not in (".zip", ".pkl"):
                continue
            if f.suffix == ".zip" and not is_valid_zip(f):
                print(f"  ⚠ Corrupt checkpoint skipped: {f.name}")
                skipped.append(f.name)
                notifier.send(
                    f"Corrupt checkpoint skipped — {stage_name}",
                    f"Checkpoint {f.name} failed integrity check and was skipped.\n"
                    f"Training will start fresh for stage {stage_name}."
                )
                continue
            shutil.copy2(f, CHECKPOINT_DIR / f.name)
            restored.append(f.name)

    if restored:
        print(f"✓ Restored {len(restored)} checkpoint files for {stage_name}: {restored}")
    else:
        print(f"⚠ No valid cloud checkpoint found for {stage_name} — starting fresh")
    if skipped:
        print(f"⚠ Skipped corrupt files: {skipped}")


# ─── Aggressive Sync ─────────────────────────────────────────────────────────

def full_sync(start_stage: int, notifier: EmailNotifier, push_fail_count: list):
    """Sync EVERYTHING to git: all checkpoints, logs, training data.

    This is the nuclear option — if VM dies right after a sync, you lose
    at most one sync interval worth of training.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M")

    # 1. Copy ALL checkpoint files to cloud_checkpoints/<stage>/
    synced_files = []
    for stg_idx in range(start_stage, 7):
        sname = STAGE_NAMES.get(stg_idx)
        if not sname:
            continue
        cloud_stage_dir = CLOUD_DIR / sname
        cloud_stage_dir.mkdir(parents=True, exist_ok=True)

        # Sync best + final only (step checkpoints stay local, gitignored)
        for pattern in [
            f"{sname}_best.zip", f"vecnormalize_{sname}_best.pkl",
            f"final_{sname}.zip", f"vecnormalize_final_{sname}.pkl",
        ]:
            for src in CHECKPOINT_DIR.glob(pattern):
                dest = cloud_stage_dir / src.name
                if not dest.exists() or src.stat().st_mtime > dest.stat().st_mtime:
                    shutil.copy2(src, dest)
                    synced_files.append(f"{sname}/{src.name}")

    # Also sync final_stage*.zip (numbered naming from train.py final save)
    for src in CHECKPOINT_DIR.glob("final_stage*.zip"):
        # Put in the right stage folder based on stage number
        dest = CLOUD_DIR / src.name
        if not dest.exists() or src.stat().st_mtime > dest.stat().st_mtime:
            shutil.copy2(src, dest)
            synced_files.append(src.name)
    for src in CHECKPOINT_DIR.glob("vecnormalize_final_stage*.pkl"):
        dest = CLOUD_DIR / src.name
        if not dest.exists() or src.stat().st_mtime > dest.stat().st_mtime:
            shutil.copy2(src, dest)
            synced_files.append(src.name)

    if synced_files:
        print(f"  ✓ Synced {len(synced_files)} checkpoint files")
    else:
        print(f"  No new checkpoint files to sync")

    # 2. Git add cloud_checkpoints only (step checkpoints + logs are gitignored)
    subprocess.run(["git", "add", "cloud_checkpoints/"], cwd=REPO_ROOT,
                   check=False, capture_output=True)

    # 3. Check if there's anything to commit
    diff = subprocess.run(["git", "diff", "--staged", "--quiet"], cwd=REPO_ROOT)
    if diff.returncode == 0:
        print(f"  No changes to commit")
        return

    # 4. Commit + push
    commit_msg = f"auto: training sync {timestamp}"
    subprocess.run(
        ["git", "commit", "-m", commit_msg],
        cwd=REPO_ROOT, check=False, capture_output=True,
    )

    result = subprocess.run(
        ["git", "push"],
        cwd=REPO_ROOT, check=False, capture_output=True,
    )

    if result.returncode == 0:
        print(f"  ✓ Pushed to GitHub @ {timestamp}")
        push_fail_count[0] = 0
    else:
        push_fail_count[0] += 1
        stderr = result.stderr.decode() if result.stderr else ""
        print(f"  ⚠ Git push failed #{push_fail_count[0]}: {stderr[:200]}")

        # Try pull --rebase then push again
        if push_fail_count[0] <= 2:
            print(f"  Attempting pull --rebase + push...")
            subprocess.run(["git", "pull", "--rebase"], cwd=REPO_ROOT,
                           check=False, capture_output=True)
            retry = subprocess.run(["git", "push"], cwd=REPO_ROOT,
                                   check=False, capture_output=True)
            if retry.returncode == 0:
                print(f"  ✓ Push succeeded after rebase")
                push_fail_count[0] = 0
                return

        if push_fail_count[0] >= 3:
            notifier.send(
                f"Git push failing #{push_fail_count[0]}",
                f"GitHub push has failed {push_fail_count[0]} consecutive times.\n\n"
                f"Checkpoints are saved locally but NOT on GitHub.\n"
                f"If the VM dies, data will be LOST.\n\n"
                f"Check: GITHUB_TOKEN, network, repo permissions."
            )


def background_sync(start_stage: int, notifier: EmailNotifier,
                    push_fail_count: list, interval: int = 300):
    """Full sync every interval seconds in a background thread."""
    while True:
        time.sleep(interval)
        try:
            print(f"\n[sync] full sync @ {time.strftime('%H:%M:%S')}")
            full_sync(start_stage, notifier, push_fail_count)
        except Exception as e:
            print(f"[sync] Error: {e}")


# ─── Hourly Status Email ─────────────────────────────────────────────────────

STATUS_FILE = REPO_ROOT / "rl_training" / "training_status.json"

def read_training_status() -> dict:
    """Read live training status written by train.py callback."""
    try:
        with open(STATUS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def hourly_status_email(notifier: EmailNotifier, start_time: float):
    """Send training status email every hour."""
    report_num = 0
    while True:
        time.sleep(3600)  # 1 hour
        report_num += 1
        status = read_training_status()
        if not status:
            continue

        elapsed_h = (time.time() - start_time) / 3600
        stage_name = status.get("stage_name", "?")
        success_rate = status.get("success_rate", 0)
        best_rate = status.get("best_success_rate", 0)
        timesteps = status.get("timesteps", 0)
        episodes = status.get("episodes", 0)
        stage_eps = status.get("stage_episodes", 0)

        subject = f"Hourly #{report_num} — {stage_name} @ {success_rate:.1%}"
        body = (
            f"LEAA Training Status Report #{report_num}\n"
            f"{'='*40}\n\n"
            f"Stage:            {status.get('stage', '?')} — {stage_name}\n"
            f"Success rate:     {success_rate:.1%}\n"
            f"Best success:     {best_rate:.1%}\n"
            f"Total timesteps:  {timesteps:,}\n"
            f"Total episodes:   {episodes:,}\n"
            f"Stage episodes:   {stage_eps:,}\n"
            f"Elapsed:          {elapsed_h:.1f}h\n"
            f"Last update:      {status.get('timestamp', '?')}\n"
        )

        notifier.send(subject, body)
        print(f"[status] Hourly report #{report_num} sent: {stage_name} @ {success_rate:.1%}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LEAA Training Session")
    parser.add_argument("--stage", type=int, required=True, choices=range(7),
                        help="Stage index (0=static_close .. 4=moving_slow, 5=wind, 6=full_dynamic)")
    parser.add_argument("--timesteps", type=int, default=15_000_000)
    parser.add_argument("--num-envs", type=int, default=4,
                        help="Parallel environments (default: 4)")
    parser.add_argument("--auto-advance", action="store_true",
                        help="Auto-advance to next stages when success threshold is met")
    parser.add_argument("--success-threshold", type=float, default=None,
                        help="Override success threshold for all stages (e.g. 0.9 for 90%%)")
    parser.add_argument("--sync-interval", type=int, default=300,
                        help="Full backup sync interval in seconds (default: 300 = 5min)")
    parser.add_argument("--gmail", type=str, default=None,
                        help="Gmail address for notifications")
    parser.add_argument("--gmail-password", type=str, default=None,
                        help="Gmail App Password (not your login password)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"],
                        help="Training device (default: cpu)")
    parser.add_argument("--max-runtime-hours", type=float, default=0,
                        help="Session runtime limit in hours (0 = no limit)")
    args = parser.parse_args()

    stage_name = STAGE_NAMES[args.stage]
    mode = "auto-advance (→ stage 6)" if args.auto_advance else "single stage only"

    print(f"\n{'='*60}")
    print(f"  LEAA Training Session")
    print(f"  Stage:          {stage_name} (stage {args.stage})")
    print(f"  Timesteps:      {args.timesteps:,}")
    print(f"  Envs:           {args.num_envs}")
    print(f"  Device:         {args.device}")
    print(f"  Mode:           {mode}")
    if args.success_threshold:
        print(f"  Threshold:      {args.success_threshold:.0%}")
    print(f"  Sync interval:  {args.sync_interval}s ({args.sync_interval // 60}min)")
    if args.max_runtime_hours > 0:
        print(f"  Max runtime:    {args.max_runtime_hours}h")
    else:
        print(f"  Max runtime:    unlimited")
    print(f"{'='*60}\n")

    notifier = EmailNotifier(args.gmail, args.gmail_password)
    push_fail_count = [0]

    # ── Restore checkpoint ──────────────────────────────────────────────────
    restore_checkpoint(stage_name, notifier)

    resume_path = CHECKPOINT_DIR / f"{stage_name}_best.zip"
    if not resume_path.exists():
        print(f"⚠ No resume checkpoint found — starting fresh for {stage_name}")
        resume_path = None

    # ── Background sync thread (aggressive: every 5 min by default) ──────
    sync_thread = threading.Thread(
        target=background_sync,
        args=(args.stage, notifier, push_fail_count, args.sync_interval),
        daemon=True,
    )
    sync_thread.start()
    print(f"✓ Background sync started (every {args.sync_interval}s)")
    print(f"  Syncs: checkpoints + logs + training data → git push")

    # ── Hourly status email thread ───────────────────────────────────────────
    session_start = time.time()
    status_thread = threading.Thread(
        target=hourly_status_email,
        args=(notifier, session_start),
        daemon=True,
    )
    status_thread.start()
    print(f"✓ Hourly status emails enabled")

    # ── Runtime watchdog ────────────────────────────────────────────────────
    watchdog = None
    if args.max_runtime_hours > 0:
        watchdog = RuntimeWatchdog(args.max_runtime_hours, notifier, stage_name)
        watchdog.start()
        print(f"✓ Runtime watchdog started "
              f"(stops at {args.max_runtime_hours - 1:.0f}h elapsed)\n")
    else:
        print("✓ No runtime limit\n")

    # ── Signal handler for sudden death ──────────────────────────────────
    # If VM is shutting down, catch SIGTERM and do one last sync
    _training_proc = [None]  # mutable ref for signal handler

    def emergency_sync(signum, frame):
        sig_name = signal.Signals(signum).name
        print(f"\n🚨 {sig_name} received — emergency sync starting...")
        notifier.send(
            f"Emergency shutdown — {sig_name}",
            f"VM sent {sig_name}. Running emergency checkpoint sync before exit."
        )
        # Kill training process so it saves checkpoint
        proc = _training_proc[0]
        if proc and proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
        # Final sync
        try:
            full_sync(args.stage, notifier, push_fail_count)
            print("✓ Emergency sync complete")
        except Exception as e:
            print(f"⚠ Emergency sync failed: {e}")
        sys.exit(1)

    signal.signal(signal.SIGTERM, emergency_sync)
    signal.signal(signal.SIGHUP, emergency_sync)

    # ── Email: training started ─────────────────────────────────────────────
    notifier.send(
        f"Training started — Stage {args.stage}: {stage_name}",
        f"Training session started.\n\n"
        f"Stage:      {stage_name} (stage {args.stage})\n"
        f"Mode:       {mode}\n"
        f"Timesteps:  {args.timesteps:,}\n"
        f"Envs:       {args.num_envs}\n"
        f"Device:     {args.device}\n"
        f"Sync:       every {args.sync_interval}s → GitHub\n"
        f"Runtime:    {'unlimited' if args.max_runtime_hours == 0 else f'{args.max_runtime_hours}h'}\n\n"
        f"{'Resuming from ' + resume_path.name if resume_path else 'Starting fresh.'}"
    )

    # ── Build training command ──────────────────────────────────────────────
    cmd = [
        "python", "rl_training/train.py",
        "--device", args.device,
        "--num-envs", str(args.num_envs),
        "--timesteps", str(args.timesteps),
        "--start-stage", str(args.stage),
    ]
    if not args.auto_advance:
        cmd += ["--max-stage", str(args.stage)]
    if args.success_threshold:
        cmd += ["--success-threshold", str(args.success_threshold)]
    if resume_path:
        cmd += ["--resume", str(resume_path)]
        print(f"▶ Resuming from: {resume_path.name}")
    else:
        print(f"▶ Starting fresh for stage {args.stage}")

    print(f"▶ Command: {' '.join(cmd)}\n")

    # ── Run training ────────────────────────────────────────────────────────
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT)
    _training_proc[0] = proc
    if watchdog:
        watchdog.register_process(proc)

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n[main] Interrupted — saving checkpoint...")
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()

    if watchdog:
        watchdog.stop()

    # ── Final sync (guaranteed) ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training ended — running FINAL full sync...")
    print(f"{'='*60}")
    full_sync(args.stage, notifier, push_fail_count)

    # ── Email: session ended ────────────────────────────────────────────────
    stopped_by_watchdog = watchdog._stopped if watchdog else False
    notifier.send(
        f"{'Session ended (runtime limit)' if stopped_by_watchdog else 'Training complete'} — {stage_name}",
        f"Stage {args.stage} ({stage_name}) training has ended.\n\n"
        f"Mode: {mode}\n"
        f"{'⏱ Stopped by runtime watchdog.' if stopped_by_watchdog else '✓ Completed normally.'}\n\n"
        f"All checkpoints + logs synced to GitHub."
    )

    print("\n✓ Done. Session complete.")


if __name__ == "__main__":
    main()
