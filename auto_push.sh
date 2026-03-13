#!/bin/bash
while true; do
    sleep 300
    cd /content/leaa
    git add -A
    if ! git diff --cached --quiet; then
        git commit -m "checkpoint: moving_slow best update"
        git push origin main 2>&1
        echo "$(date): Pushed." >> /content/leaa/auto_commit.log
    fi
done
