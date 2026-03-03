#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/git_commit.sh \"commit message\""
  exit 1
fi

msg="$1"

git status -sb
git add -A
git commit -m "$msg"
