#!/bin/bash
# Refresh this private monorepo from the working folders and push.
# Usage: ./sync_from_working_folders.sh ["commit message"]
set -e
cd "$(dirname "$0")"
W=/home/ailab/VM/Lav
for r in road-masquerade-floor mavlink-cross-campaign; do
  b=$(git -C "$W/$r" rev-parse --abbrev-ref HEAD)
  git subtree pull -q --prefix="$r" "$W/$r" "$b" -m "Sync $r" 2>&1 | grep -v "^From\|FETCH_HEAD\|Already up to date" || true
done
# Top-level documents (the GALAXY PDF is deliberately left out: licensed IEEE download)
cp -f "$W"/ICCE2027_SS02_Project_Blueprints.docx "$W"/outreach_drafts.md "$W"/Paper*_DRAFT.pdf .
# Project 1 keeps results/ and data/ out of its own repo; bring the small, useful parts here
P1="$W/road-masquerade-floor"
rsync -a --delete --exclude 'nll/' "$P1/results/" road-masquerade-floor/results/
mkdir -p road-masquerade-floor/data/raw/crysys
cp -f "$P1/data/raw/pdl.sh" road-masquerade-floor/data/raw/pdl.sh
cp -f "$P1/data/raw/crysys/pdl.sh" road-masquerade-floor/data/raw/crysys/pdl.sh
for r in road-masquerade-floor mavlink-cross-campaign; do [ -f "$W/$r/paper/main.pdf" ] && cp -f "$W/$r/paper/main.pdf" "$r/paper/main.pdf"; done
git add -A .
git add -f road-masquerade-floor/results road-masquerade-floor/data/raw/pdl.sh road-masquerade-floor/data/raw/crysys/pdl.sh \
           road-masquerade-floor/paper/main.pdf mavlink-cross-campaign/paper/main.pdf
# safety net: refuse anything GitHub would reject or that should never be uploaded
big=$(git diff --cached --name-only -z | xargs -0 -r du -k 2>/dev/null | awk '$1 > 50000 {print $2}')
[ -n "$big" ] && { echo "REFUSING: staged files over 50 MB:"; echo "$big"; exit 1; }
git diff --cached --quiet || git commit -q -m "${1:-Sync results, papers and top-level documents}"
if git remote get-url origin >/dev/null 2>&1; then git push -q origin main && echo "pushed to $(git remote get-url origin)"; else echo "no remote yet: run ./push_to_github.sh after 'gh auth login'"; fi
