#!/bin/bash
cd /content/leaa
git add -A
if ! git diff --cached --quiet; then
    git commit -m "auto: training checkpoint $(date +%Y-%m-%d\ %H:%M)"
    git push origin main 2>&1
    echo "$(date): Committed and pushed." >> /content/leaa/auto_commit.log
else
    echo "$(date): No changes to commit." >> /content/leaa/auto_commit.log
fi
