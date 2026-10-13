#!/bin/bash
# One-time: create the PRIVATE GitHub repo and push. Run after `gh auth login`.
# Usage: ./push_to_github.sh [repo-name]     (default: icce2027-ids-papers)
set -e
cd "$(dirname "$0")"
NAME="${1:-icce2027-ids-papers}"
gh auth status >/dev/null 2>&1 || { echo "Not logged in. Run:  gh auth login   (choose GitHub.com, HTTPS, login with a web browser)"; exit 1; }
USER=$(gh api user -q .login)
if git remote get-url origin >/dev/null 2>&1; then
  echo "remote already set: $(git remote get-url origin)"
else
  gh repo create "$NAME" --private --source=. --remote=origin --description "ICCE 2027: CAN masquerade shortcut audit and MAVLink cross-campaign IDS (private working repo)"
fi
gh auth setup-git 2>/dev/null || true
git push -u origin main
# verify it really is private before reporting success
PRIV=$(gh api "repos/$USER/$NAME" -q .private)
echo "repo: https://github.com/$USER/$NAME   private: $PRIV"
[ "$PRIV" = "true" ] || { echo "WARNING: repository is NOT private. Fix with: gh api -X PATCH repos/$USER/$NAME -f private=true"; exit 1; }
