#!/bin/bash
set -e

echo ">>> Pushing training results to GitHub..."

# Stage training artifacts
git add -f rl_training/checkpoints/ rl_training/logs/ training_output.log 2>/dev/null || true
git add -A

git commit -m "Phase 1 training results - $(date '+%Y-%m-%d %H:%M')"
git push origin main

echo ""
echo "✅ Results pushed to GitHub!"
echo ""
echo "On your MacBook, run:"
echo "  cd ~/Desktop/gig_projects/AI_Projects/LEAA"
echo "  git pull origin main"
echo "  source .venv/bin/activate"
echo '  PYTHONPATH=. python rl_training/evaluate.py --model rl_training/checkpoints/final_stage0.zip --visualize'
echo ""
