#!/bin/bash
set -e

echo "============================================"
echo " LEAA — Linux Setup + Training"
echo " Language Embedded Agent Actions"
echo "============================================"
echo ""

# ---------- System Dependencies ----------
echo ">>> Installing system dependencies..."
sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip git

# ---------- Virtual Environment ----------
echo ""
echo ">>> Creating Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip "setuptools<81" wheel

# ---------- PyTorch (CPU-only, saves ~2GB vs CUDA) ----------
echo ""
echo ">>> Installing PyTorch (CPU-only)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# ---------- Other Dependencies ----------
echo ""
echo ">>> Installing remaining dependencies..."
pip install numpy scipy matplotlib gymnasium "stable-baselines3[extra]" pyyaml tensorboard tqdm rich

# ---------- Verify ----------
echo ""
echo ">>> Verifying installation..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
import stable_baselines3
print(f'SB3: {stable_baselines3.__version__}')
import gymnasium
print(f'Gymnasium: {gymnasium.__version__}')
print('All imports OK!')
"

# ---------- Train ----------
echo ""
echo "============================================"
echo " Starting Training (2M steps)"
echo " This will take ~8-15 hours on CPU"
echo " Progress bar will show below"
echo "============================================"
echo ""

PYTHONPATH=. python rl_training/train.py --device cpu 2>&1 | tee training_output.log

echo ""
echo "============================================"
echo " Training Complete!"
echo "============================================"
echo ""
echo "Checkpoints saved to: rl_training/checkpoints/"
echo "TensorBoard logs:     rl_training/logs/"
echo ""
echo "View TensorBoard:     tensorboard --logdir rl_training/logs/"
echo "Push results:         ./push_results.sh"
echo ""
