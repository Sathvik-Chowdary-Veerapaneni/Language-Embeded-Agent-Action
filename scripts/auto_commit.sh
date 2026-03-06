#!/bin/bash

REPO_DIR="/Users/sathvikchowdaryveerapaneni/Desktop/gig_projects/AI_Projects/LEAA"
cd "$REPO_DIR" || exit 1

# Only commit if there are changes
if git diff --quiet && git diff --cached --quiet && [ -z "$(git ls-files --others --exclude-standard)" ]; then
    echo "$(date): No changes to commit."
    exit 0
fi

git add -A
git commit -m "auto: training checkpoint $(date '+%Y-%m-%d %H:%M')"
git push
echo "$(date): Committed and pushed."
