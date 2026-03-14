"""
LEAA Colab Training Session Script
===================================
Handles one training stage in a Colab session:
  - Restores checkpoint from cloud_checkpoints/<stage>/
  - Validates checkpoint integrity before loading
  - Runs training (locked to that stage via --max-stage)
  - Syncs best checkpoint to cloud_checkpoints/<stage>/ every 30 min
  - Pushes to GitHub on each sync; emails alert after 3 consecutive failures
  - Runtime watchdog: emails 1h warning, gracefully stops training 1h before limit
  - Emails on: training start, runtime warning, training end

Usage (in Colab):
    python scripts/colab_train.py --stage 3 --timesteps 15000000 --num-envs 4

    # With email notifications (recommended):
    python scripts/colab_train.py --stage 3 --timesteps 15000000 --num-envs 4 \\
        --gmail you@gmail.com --gmail-password <app_password>

Stage map:
    3 = static_far
    4 = moving_slow
    5 = wind
    6 = full_dynamic
"""

import argparse
import os
import shutil
import signal
import smtplib
import subprocess
import threading
import time
import zipfile
from email.mime.text import MIMEText
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


# ─── Email ────────────────────────────────────────────────────────────────────

class EmailNotifier:
    """Send Gmail notifications via SMTP App Password."""

    def __init__(self, gmail_address: str = None, app_password: str = None):
        self.address = gmail_address
        self.password = app_password
        self.enabled = bool(gmail_address and app_password)
        if self.enabled:
            print(f"✓ Email notifications enabled → {gmail_address}")
        else:
            print("⚠ Email notifications disabled (no credentials provided)")

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
    - Sends SIGINT to the training process at (limit - 1h) for graceful stop + checkpoint save
    """

    def __init__(self, max_runtime_hours: float, notifier: EmailNotifier, stage_name: str):
        self.max_seconds = max_runtime_hours * 3600
        self.buffer_seconds = 3600          # 1h buffer before hard limit
        self.warning_lead = 300             # warn 5 min before buffer kicks in
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

            # Warning: 5 min before the 1h buffer kicks in
            if not self._warned and remaining <= (self.buffer_seconds + self.warning_lead):
                self._warned = True
                print(f"\n⚠ Runtime warning: ~1h until graceful stop "
                      f"(elapsed {self._fmt(elapsed)}, remaining {self._fmt(remaining)})")
                self.notifier.send(
                    f"Runtime warning — {self.stage_name} (~1h left)",
                    f"Colab session is approaching its time limit.\n\n"
                    f"Stage:    {self.stage_name}\n"
                    f"Elapsed:  {self._fmt(elapsed)}\n"
                    f"Remaining until stop: ~{self._fmt(remaining)}\n\n"
                    f"Training will be stopped gracefully in ~1 hour and the\n"
                    f"latest checkpoint will be synced to GitHub.\n\n"
                    f"Re-open the notebook to resume from where it left off."
                )

            # Stop: 1h before hard limit
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
                    f"Training will start fresh for stage {stage_name}.\n\n"
                    f"You may want to delete the corrupt file from cloud_checkpoints/{stage_name}/ "
                    f"and retrain from the previous stage."
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


def sync_checkpoint(stage_name: str, notifier: EmailNotifier, push_fail_count: list):
    """Copy best checkpoint → cloud_checkpoints/<stage>/ and push to GitHub."""
    cloud_stage_dir = CLOUD_DIR / stage_name
    cloud_stage_dir.mkdir(parents=True, exist_ok=True)

    synced = []
    for pattern in [
        f"{stage_name}_best.zip", f"vecnormalize_{stage_name}_best.pkl",
        f"final_{stage_name}.zip", f"vecnormalize_final_{stage_name}.pkl",
    ]:
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

    subprocess.run(["git", "add", "cloud_checkpoints/"], cwd=REPO_ROOT, check=False)
    diff = subprocess.run(["git", "diff", "--staged", "--quiet"], cwd=REPO_ROOT)
    if diff.returncode != 0:
        subprocess.run(
            ["git", "commit", "-m", f"checkpoint: {stage_name} best update"],
            cwd=REPO_ROOT, check=False,
        )
        result = subprocess.run(["git", "push"], cwd=REPO_ROOT, check=False)
        if result.returncode == 0:
            print(f"✓ Pushed {stage_name} checkpoint to GitHub")
            push_fail_count[0] = 0  # reset on success
        else:
            push_fail_count[0] += 1
            print(f"⚠ Git push failed (failure #{push_fail_count[0]}) — retrying next sync")
            if push_fail_count[0] >= 3:
                notifier.send(
                    f"Git push failing — {stage_name} (#{push_fail_count[0]})",
                    f"GitHub push has failed {push_fail_count[0]} consecutive times.\n\n"
                    f"Stage: {stage_name}\n\n"
                    f"Checkpoints are saved on the VM but NOT on GitHub.\n"
                    f"If the session expires, they will be LOST.\n\n"
                    f"Possible causes:\n"
                    f"  - GITHUB_TOKEN expired → update in Colab Secrets\n"
                    f"  - Network issue on the VM\n"
                    f"  - Repository permission problem\n\n"
                    f"Fix the token and re-run Cell 2 to re-embed it in the remote URL."
                )


def background_sync(stage_name: str, notifier: EmailNotifier,
                    push_fail_count: list, interval: int = 1800):
    """Sync checkpoint every interval seconds in a background thread."""
    while True:
        time.sleep(interval)
        try:
            print(f"\n[sync] {stage_name} @ {time.strftime('%H:%M:%S')}")
            sync_checkpoint(stage_name, notifier, push_fail_count)
        except Exception as e:
            print(f"[sync] Error: {e}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LEAA Colab Training Session")
    parser.add_argument("--stage", type=int, required=True, choices=[3, 4, 5, 6],
                        help="Stage index (3=static_far, 4=moving_slow, 5=wind, 6=full_dynamic)")
    parser.add_argument("--timesteps", type=int, default=15_000_000)
    parser.add_argument("--num-envs", type=int, default=4,
                        help="Parallel environments (default: 4 for Colab 2-vCPU)")
    parser.add_argument("--sync-interval", type=int, default=1800,
                        help="Checkpoint sync interval in seconds (default: 1800 = 30min)")
    parser.add_argument("--gmail", type=str, default=None,
                        help="Gmail address for notifications")
    parser.add_argument("--gmail-password", type=str, default=None,
                        help="Gmail App Password (not your login password)")
    parser.add_argument("--max-runtime-hours", type=float, default=11.0,
                        help="Session runtime limit in hours — watchdog stops training 1h before this "
                             "(default: 11h, leaving 1h buffer in a 12h Colab Pro session)")
    args = parser.parse_args()

    stage_name = STAGE_NAMES[args.stage]

    print(f"\n{'='*55}")
    print(f"  LEAA Colab Training: {stage_name} (stage {args.stage})")
    print(f"  Timesteps:      {args.timesteps:,}")
    print(f"  Envs:           {args.num_envs}")
    print(f"  Sync interval:  {args.sync_interval // 60} min")
    print(f"  Max runtime:    {args.max_runtime_hours}h  (stops 1h before = at {args.max_runtime_hours - 1}h)")
    print(f"{'='*55}\n")

    notifier = EmailNotifier(args.gmail, args.gmail_password)
    push_fail_count = [0]  # list so background thread can mutate it

    # ── Restore checkpoint ──────────────────────────────────────────────────
    restore_checkpoint(stage_name, notifier)

    resume_path = CHECKPOINT_DIR / f"{stage_name}_best.zip"
    if not resume_path.exists():
        print(f"⚠ No resume checkpoint found — starting fresh for {stage_name}")
        resume_path = None

    # ── Background sync thread ──────────────────────────────────────────────
    sync_thread = threading.Thread(
        target=background_sync,
        args=(stage_name, notifier, push_fail_count, args.sync_interval),
        daemon=True,
    )
    sync_thread.start()
    print(f"✓ Background sync started (every {args.sync_interval // 60} min)")

    # ── Runtime watchdog ────────────────────────────────────────────────────
    watchdog = RuntimeWatchdog(args.max_runtime_hours, notifier, stage_name)
    watchdog.start()
    print(f"✓ Runtime watchdog started "
          f"(warning at {args.max_runtime_hours - 1:.0f}h, hard stop at {args.max_runtime_hours - 1:.0f}h elapsed)\n")

    # ── Email: training started ─────────────────────────────────────────────
    notifier.send(
        f"Training started — Stage {args.stage}: {stage_name}",
        f"A new training session has started on Colab.\n\n"
        f"Stage:          {stage_name} (stage {args.stage})\n"
        f"Timesteps:      {args.timesteps:,}\n"
        f"Parallel envs:  {args.num_envs}\n"
        f"Checkpoint sync: every {args.sync_interval // 60} min → GitHub\n"
        f"Max runtime:    {args.max_runtime_hours}h "
        f"(graceful stop at {args.max_runtime_hours - 1:.0f}h)\n\n"
        f"{'Resuming from last checkpoint.' if resume_path else 'Starting fresh (no prior checkpoint).'}"
    )

    # ── Build training command ──────────────────────────────────────────────
    cmd = [
        "python", "rl_training/train.py",
        "--device", "cuda",
        "--num-envs", str(args.num_envs),
        "--timesteps", str(args.timesteps),
        "--start-stage", str(args.stage),
        "--max-stage", str(args.stage),
    ]
    if resume_path:
        cmd += ["--resume", str(resume_path)]
        print(f"▶ Resuming from: {resume_path.name}")
    else:
        print(f"▶ Starting fresh for stage {args.stage}")

    print(f"▶ Command: {' '.join(cmd)}\n")

    # ── Run training via Popen so watchdog can stop it ──────────────────────
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT)
    watchdog.register_process(proc)

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n[main] Interrupted — waiting for training process to finish saving...")
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()

    watchdog.stop()

    # ── Final sync ──────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Training ended — running final checkpoint sync...")
    print(f"{'='*55}")
    sync_checkpoint(stage_name, notifier, push_fail_count)

    # ── Email: session ended ────────────────────────────────────────────────
    stopped_by_watchdog = watchdog._stopped
    notifier.send(
        f"{'Session ended (runtime limit)' if stopped_by_watchdog else 'Training complete'} — {stage_name}",
        f"Stage {args.stage} ({stage_name}) training session has ended.\n\n"
        f"{'⏱ Stopped by runtime watchdog (1h buffer).' if stopped_by_watchdog else '✓ Completed normally.'}\n\n"
        f"Latest checkpoint synced to GitHub.\n\n"
        f"{'→ Re-open the notebook and run all cells to resume.' if stopped_by_watchdog else '→ You can now start the next stage.'}"
    )

    print("\n✓ Done. Session complete.")


if __name__ == "__main__":
    main()
