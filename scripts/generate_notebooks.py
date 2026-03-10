"""Generate 4 Colab training notebooks, one per stage."""
import json

STAGES = [
    {"num": 3, "name": "static_far",   "timesteps": 15_000_000, "target": "85%"},
    {"num": 4, "name": "moving_slow",  "timesteps": 15_000_000, "target": "70%"},
    {"num": 5, "name": "wind",         "timesteps": 20_000_000, "target": "55%"},
    {"num": 6, "name": "full_dynamic", "timesteps": 25_000_000, "target": "35%"},
]

REPO = "Sathvik-Chowdary-Veerapaneni/Language-Embeded-Agent-Action"

def make_notebook(stage):
    n = stage["num"]
    name = stage["name"]
    ts = stage["timesteps"]
    target = stage["target"]

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# LEAA Training — Stage {n}: `{name}`\n",
                f"**Target accuracy:** {target}  |  **Timesteps:** {ts:,}\n\n",
                "### Setup Instructions\n",
                "1. Go to **Runtime → Change runtime type → T4 GPU**\n",
                "2. Add your GitHub PAT to Colab Secrets:\n",
                "   - Left sidebar → 🔑 Secrets → `GITHUB_TOKEN`\n",
                "3. Run all cells in order\n",
                "4. When session expires, re-open this notebook and run all cells — it auto-resumes\n",
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 1: Verify GPU\n",
                "import torch\n",
                "print(f'CUDA available: {torch.cuda.is_available()}')\n",
                "if torch.cuda.is_available():\n",
                "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n",
                "else:\n",
                "    raise RuntimeError('No GPU detected! Go to Runtime → Change runtime type → T4 GPU')",
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 2: Authenticate & Clone repo\n",
                "from google.colab import userdata\n",
                "import os, subprocess\n",
                "\n",
                "TOKEN = userdata.get('GITHUB_TOKEN')\n",
                f"REPO = '{REPO}'\n",
                "CLONE_URL = f'https://{{TOKEN}}@github.com/{{REPO}}.git'\n",
                "\n",
                "if not os.path.exists('/content/leaa'):\n",
                "    subprocess.run(['git', 'clone', CLONE_URL, '/content/leaa'], check=True)\n",
                "else:\n",
                "    subprocess.run(['git', 'pull'], cwd='/content/leaa', check=True)\n",
                "\n",
                "# Configure git identity for pushes\n",
                "subprocess.run(['git', 'config', 'user.email', 'colab@leaa.bot'], cwd='/content/leaa')\n",
                "subprocess.run(['git', 'config', 'user.name', 'Colab Training Bot'], cwd='/content/leaa')\n",
                "\n",
                "# Embed token in remote URL so pushes work without interactive auth\n",
                "subprocess.run(['git', 'remote', 'set-url', 'origin', CLONE_URL], cwd='/content/leaa')\n",
                "print('✓ Repo ready at /content/leaa')",
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 3: Install dependencies\n",
                "%cd /content/leaa\n",
                "!pip install -q -r requirements.txt\n",
                "print('✓ Dependencies installed')",
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 4: Run training\n",
                "# This cell runs for the full session (~11 hrs).\n",
                "# Checkpoints auto-sync to GitHub every 30 min.\n",
                "# If session expires, re-run all cells — training resumes from last checkpoint.\n",
                "%cd /content/leaa\n",
                f"!python scripts/colab_train.py --stage {n} --timesteps {ts} --num-envs 4",
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 5: (Optional) Evaluate this stage after training\n",
                "%cd /content/leaa\n",
                f"!python rl_training/evaluate.py \\\\\n",
                f"    --model rl_training/checkpoints/{name}_best.zip \\\\\n",
                f"    --vecnorm rl_training/checkpoints/vecnormalize_{name}_best.pkl \\\\\n",
                f"    --stage {name} \\\\\n",
                f"    --episodes 200",
            ]
        },
    ]

    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
        },
        "cells": cells,
    }
    return notebook


import os
out_dir = "/Users/sathvikchowdaryveerapaneni/Desktop/gig_projects/AI_Projects/LEAA/colab"
os.makedirs(out_dir, exist_ok=True)

for stage in STAGES:
    nb = make_notebook(stage)
    path = f"{out_dir}/stage{stage['num']}_{stage['name']}.ipynb"
    with open(path, "w") as f:
        json.dump(nb, f, indent=2)
    print(f"✓ Created: {path}")
