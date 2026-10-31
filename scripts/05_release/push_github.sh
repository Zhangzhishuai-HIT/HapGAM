#!/bin/bash
# Copy a code archive into the local clone of the GitHub repository (release/github), commit and push.
#   bash scripts/05_release/push_github.sh [date] [commit message]   # default: newest release/HapGAM_code_*.tar.gz
# The clone is created on first use, with the remote given in HAPGAM_GIT_REMOTE.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repository root
A=$(ls release/HapGAM_code_${1:-*}.tar.gz | sort | tail -1)
D=$(basename "$A" .tar.gz); D=${D#HapGAM_code_}
G=release/github
if [ ! -d "$G/.git" ]; then
  mkdir -p "$G"
  (cd "$G" && git init -q && git remote add origin "${HAPGAM_GIT_REMOTE:?set HAPGAM_GIT_REMOTE for the first push}")
fi
cd "$G"
git rm -r -q --ignore-unmatch . > /dev/null   # files removed from the archive are removed from the repository
tar xzf "../../$A" --strip-components=1
git add -A .
if git rev-parse -q --verify HEAD > /dev/null && git diff --cached --quiet; then
  echo "no changes against $(git log --oneline -1)"; exit 0
fi
git commit -q -m "${2:-HapGAM code (archive $D)}"
git branch -M main
git push -q -u origin main
git log --oneline -1
