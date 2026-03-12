"""Generate 4 Colab training notebooks, one per stage."""
import json
import os

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
        # ── Markdown: instructions ────────────────────────────────────────
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# LEAA Training — Stage {n}: `{name}`\n",
                f"**Target accuracy:** {target}  |  **Timesteps:** {ts:,}\n\n",
                "### Setup Instructions\n",
                "1. **Runtime → Change runtime type → T4 GPU**\n",
                "2. Fill in your credentials in **Cell 0** below\n",
                "3. Run all cells in order\n",
                "4. When session expires, re-open and run all cells — training auto-resumes\n",
            ],
        },
        # ── Cell 0: Credentials ───────────────────────────────────────────
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 0: Credentials — fill in your values here\n",
                "# DO NOT commit this notebook with real values filled in\n",
                "import os\n",
                "\n",
                "os.environ['GITHUB_TOKEN']       = 'your_github_pat_here'\n",
                "os.environ['GMAIL_ADDRESS']      = 'your@gmail.com'       # optional\n",
                "os.environ['GMAIL_APP_PASSWORD'] = 'your_app_password'    # optional\n",
                "\n",
                "print('✓ Credentials set')",
            ],
        },
        # ── Cell 1: Verify GPU ────────────────────────────────────────────
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
                "    raise RuntimeError('No GPU — go to Runtime → Change runtime type → T4 GPU')",
            ],
        },
        # ── Cell 2: Auth & clone repo ─────────────────────────────────────
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 2: Authenticate & clone repo\n",
                "from google.colab import userdata\n",
                "import os, subprocess\n",
                "\n",
                "TOKEN = userdata.get('GITHUB_TOKEN')\n",
                f"REPO = '{REPO}'\n",
                "CLONE_URL = f'https://{TOKEN}@github.com/{REPO}.git'\n",
                "\n",
                "if not os.path.exists('/content/leaa'):\n",
                "    subprocess.run(['git', 'clone', CLONE_URL, '/content/leaa'], check=True)\n",
                "else:\n",
                "    subprocess.run(['git', 'pull'], cwd='/content/leaa', check=True)\n",
                "\n",
                "subprocess.run(['git', 'config', 'user.email', 'colab@leaa.bot'], cwd='/content/leaa')\n",
                "subprocess.run(['git', 'config', 'user.name', 'Colab Training Bot'], cwd='/content/leaa')\n",
                "subprocess.run(['git', 'remote', 'set-url', 'origin', CLONE_URL], cwd='/content/leaa')\n",
                "print('✓ Repo ready at /content/leaa')",
            ],
        },
        # ── Cell 3: Email credentials (optional) ─────────────────────────
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 3: Load email credentials (optional)\n",
                "# Skip this cell if you don't want email notifications.\n",
                "from google.colab import userdata\n",
                "\n",
                "try:\n",
                "    GMAIL_ADDRESS = userdata.get('GMAIL_ADDRESS')\n",
                "    GMAIL_APP_PASSWORD = userdata.get('GMAIL_APP_PASSWORD')\n",
                "    print(f'✓ Email notifications enabled → {GMAIL_ADDRESS}')\n",
                "except Exception:\n",
                "    GMAIL_ADDRESS = None\n",
                "    GMAIL_APP_PASSWORD = None\n",
                "    print('⚠ No email credentials found — notifications disabled')\n",
                "    print('  Add GMAIL_ADDRESS + GMAIL_APP_PASSWORD to Colab Secrets to enable')",
            ],
        },
        # ── Cell 4: Install dependencies ──────────────────────────────────
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 4: Install dependencies\n",
                "%cd /content/leaa\n",
                "!pip install -q -r requirements.txt\n",
                "print('✓ Dependencies installed')",
            ],
        },
        # ── Cell 5: Run training ──────────────────────────────────────────
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 5: Run training\n",
                "# Runs for up to 11h. Checkpoints sync to GitHub every 30 min.\n",
                "# Runtime watchdog emails a warning at 10h and stops training at 11h\n",
                "# so the VM has 1h to finish saving before Colab reclaims it.\n",
                "# If the session expires, re-run all cells — training resumes from last checkpoint.\n",
                "%cd /content/leaa\n",
                "import os\n",
                "\n",
                f"cmd = 'python scripts/colab_train.py --stage {n} --timesteps {ts} --num-envs 4 --max-runtime-hours 11'\n",
                "\n",
                "# Append email args if credentials are available\n",
                "if 'GMAIL_ADDRESS' in dir() and GMAIL_ADDRESS:\n",
                "    cmd += f' --gmail {GMAIL_ADDRESS} --gmail-password {GMAIL_APP_PASSWORD}'\n",
                "\n",
                "print(f'Running: {cmd}')\n",
                "os.system(cmd)",
            ],
        },
        # ── Cell 6: Evaluate (optional) ───────────────────────────────────
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Cell 6: (Optional) Evaluate this stage after training\n",
                "%cd /content/leaa\n",
                f"!python rl_training/evaluate.py \\\\\n",
                f"    --model rl_training/checkpoints/{name}_best.zip \\\\\n",
                f"    --vecnorm rl_training/checkpoints/vecnormalize_{name}_best.pkl \\\\\n",
                f"    --stage {name} \\\\\n",
                f"    --episodes 200",
            ],
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


out_dir = os.path.join(os.path.dirname(__file__), "..", "colab")
os.makedirs(out_dir, exist_ok=True)

for stage in STAGES:
    nb = make_notebook(stage)
    path = os.path.join(out_dir, f"stage{stage['num']}_{stage['name']}.ipynb")
    with open(path, "w") as f:
        json.dump(nb, f, indent=2)
    print(f"✓ Created: {path}")
