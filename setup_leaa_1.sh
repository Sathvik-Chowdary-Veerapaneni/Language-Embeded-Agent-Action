#!/bin/bash
# ============================================
# LEAA - Language Embedded Agent Actions
# One-shot setup script for M1 MacBook Pro
# ============================================
# Run from: ~/Desktop/gig_projects/AI_Projects/LEAA
# Usage: chmod +x setup_leaa.sh && ./setup_leaa.sh
# Cleanup: ./cleanup.sh (nukes everything)
# ============================================

set -e

PROJECT_ROOT="/Users/sathvikchowdaryveerapaneni/Desktop/gig_projects/AI_Projects/LEAA"
echo "============================================"
echo "  LEAA Project Setup"
echo "  Root: $PROJECT_ROOT"
echo "============================================"

# --- 1. Project directory structure ---
echo "[1/5] Creating project structure..."
mkdir -p "$PROJECT_ROOT"/{
  unity_project,
  rl_training/{configs,checkpoints,logs,envs},
  language_layer/{prompts,grounding},
  physics_engine,
  shared/{models,data,scripts},
  notebooks,
  tests
}

# --- 2. Python virtual environment ---
echo "[2/5] Creating Python venv..."
python3 -m venv "$PROJECT_ROOT/.venv"
source "$PROJECT_ROOT/.venv/bin/activate"

# --- 3. Upgrade pip ---
echo "[3/5] Upgrading pip..."
pip install --upgrade pip setuptools wheel

# --- 4. Install dependencies ---
echo "[4/5] Installing dependencies..."
pip install \
  torch torchvision torchaudio \
  numpy scipy matplotlib \
  gymnasium \
  stable-baselines3[extra] \
  mlagents==1.1.0 \
  mlagents-envs==1.1.0 \
  protobuf==3.20.3 \
  grpcio \
  pybullet \
  ollama \
  anthropic \
  pyyaml \
  tensorboard \
  jupyter \
  tqdm \
  rich

# --- 5. Verify MPS (Apple Silicon GPU) ---
echo "[5/5] Verifying PyTorch MPS backend..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
if torch.backends.mps.is_available():
    x = torch.randn(3, 3, device='mps')
    print(f'MPS tensor test: PASSED')
else:
    print('WARNING: MPS not available, will fallback to CPU')
"

# --- Create .env file ---
cat > "$PROJECT_ROOT/.env" << 'ENVFILE'
# LEAA Environment Config
ANTHROPIC_API_KEY=your_key_here
PROJECT_ROOT=.
DEVICE=mps
UNITY_PROJECT_PATH=./unity_project
CHECKPOINT_DIR=./rl_training/checkpoints
LOG_DIR=./rl_training/logs
ENVFILE

# --- Create .gitignore ---
cat > "$PROJECT_ROOT/.gitignore" << 'GITIGNORE'
.venv/
__pycache__/
*.pyc
.env
*.onnx
*.pt
*.pth
rl_training/checkpoints/
rl_training/logs/
unity_project/Library/
unity_project/Temp/
unity_project/Logs/
unity_project/obj/
.DS_Store
*.meta
notebooks/.ipynb_checkpoints/
GITIGNORE

# --- Create cleanup script ---
cat > "$PROJECT_ROOT/cleanup.sh" << 'CLEANUP'
#!/bin/bash
# ============================================
# LEAA FULL CLEANUP - Nukes everything
# ============================================
echo "WARNING: This will delete the entire LEAA project environment."
echo "Project: $(pwd)"
read -p "Type 'FLUSH' to confirm: " confirm
if [ "$confirm" = "FLUSH" ]; then
    deactivate 2>/dev/null || true
    rm -rf .venv
    rm -rf rl_training/checkpoints/*
    rm -rf rl_training/logs/*
    rm -rf unity_project/Library
    rm -rf unity_project/Temp
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    echo "Flushed. Project structure kept, all generated artifacts removed."
    echo "To fully nuke: rm -rf $(pwd)"
else
    echo "Aborted."
fi
CLEANUP
chmod +x "$PROJECT_ROOT/cleanup.sh"

# --- Create activation shortcut ---
cat > "$PROJECT_ROOT/activate.sh" << 'ACTIVATE'
#!/bin/bash
# Quick activate: source activate.sh
source "$(dirname "$0")/.venv/bin/activate"
echo "LEAA env activated. Python: $(which python3)"
echo "Device: $(python3 -c 'import torch; print("mps" if torch.backends.mps.is_available() else "cpu")')"
ACTIVATE
chmod +x "$PROJECT_ROOT/activate.sh"

echo ""
echo "============================================"
echo "  LEAA Setup Complete!"
echo "============================================"
echo ""
echo "  Activate env:   source activate.sh"
echo "  Cleanup:         ./cleanup.sh"
echo "  Full nuke:       rm -rf $(pwd)"
echo ""
echo "  Project structure:"
find "$PROJECT_ROOT" -maxdepth 2 -type d | grep -v ".venv" | grep -v "__pycache__" | sed "s|$PROJECT_ROOT|.|g" | sort
echo ""
echo "============================================"
