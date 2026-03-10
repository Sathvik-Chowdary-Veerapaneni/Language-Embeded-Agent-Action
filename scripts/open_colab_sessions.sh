#!/bin/bash
# Opens all 4 LEAA training notebooks in Colab (browser tabs)
# After opening: Runtime → T4 GPU → Run All in each tab

REPO="Sathvik-Chowdary-Veerapaneni/Language-Embeded-Agent-Action"
BASE="https://colab.research.google.com/github/${REPO}/blob/main/colab"

echo "Opening 4 Colab training sessions..."

open "${BASE}/stage3_static_far.ipynb"
sleep 1
open "${BASE}/stage4_moving_slow.ipynb"
sleep 1
open "${BASE}/stage5_wind.ipynb"
sleep 1
open "${BASE}/stage6_full_dynamic.ipynb"

echo ""
echo "✓ 4 tabs opened. For each tab:"
echo "  1. Runtime → Change runtime type → T4 GPU"
echo "  2. Runtime → Run all"
echo ""
echo "Monitor progress locally:"
echo "  python scripts/monitor_training.py --watch"
