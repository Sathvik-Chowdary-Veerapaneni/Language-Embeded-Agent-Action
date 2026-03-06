#!/usr/bin/env bash
# LEAA v4 — Overnight dynamic-stages training run
# Resumes from v3's final_stage3.zip (static_far complete) and trains stages 4-6:
#   Stage 4: moving_slow   (~3.0M steps, weight=3)
#   Stage 5: wind          (~4.0M steps, weight=4)
#   Stage 6: full_dynamic  (~5.0M steps, weight=5)
# Total: 12M timesteps
#
# Run from LEAA project root:
#   bash run_v4.sh
# Or with nohup for true overnight:
#   nohup bash run_v4.sh > training_output_v4.log 2>&1 &
#   tail -f training_output_v4.log

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
source .venv/bin/activate

echo "=============================="
echo " LEAA v4 — Dynamic Stage Run"
echo " $(date)"
echo "=============================="
echo ""

# Sanity check — confirm resume checkpoint exists
RESUME="rl_training/checkpoints/final_stage3.zip"
VECNORM="rl_training/checkpoints/vecnormalize_final_stage3.pkl"

if [ ! -f "$RESUME" ]; then
    echo "ERROR: checkpoint not found: $RESUME"
    exit 1
fi
if [ ! -f "$VECNORM" ]; then
    echo "ERROR: VecNormalize stats not found: $VECNORM"
    exit 1
fi

echo "Resume checkpoint : $RESUME"
echo "VecNormalize stats: $VECNORM"
echo ""

python -u rl_training/train.py \
    --resume "$RESUME" \
    --start-stage 4 \
    --timesteps 12000000 \
    --num-envs 8 \
    --device cpu \
    2>&1 | tee training_output_v4.log

echo ""
echo "=============================="
echo " v4 training complete"
echo " $(date)"
echo "=============================="
