#!/bin/bash

git add .

msg=${1:-"update at $(date +'%Y-%m-%d %H:%M:%S')"}

if ! git diff --cached --quiet; then
    echo "Changes to be committed:"
    git diff --cached --stat
    git commit -m "$msg" && git push origin main
    echo "Changes committed and pushed: $msg"
else
    echo "No changes to commit."
fi
