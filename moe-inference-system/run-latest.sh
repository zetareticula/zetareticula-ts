#!/bin/bash
# Auto-deploy latest commit and keep last good version via PM2
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

echo "🔄 Pulling latest code …"
if ! git pull --rebase --autostash origin main; then
  echo "⚠️  Git pull failed, aborting deploy" >&2
  exit 1
fi

# Wait for GitHub CI to succeed on this commit
LATEST_SHA=$(git rev-parse HEAD)
OWNER="zetareticula"
REPO="zetareticula-ts"

if [[ -z "${GITHUB_TOKEN:-}" ]]; then
  echo "⚠️  GITHUB_TOKEN not set; skipping CI status check." >&2
else
  echo "⏳ Waiting for GitHub CI checks to succeed for commit $LATEST_SHA …"
  for i in {1..20}; do
    STATUS=$(curl -s -H "Authorization: token $GITHUB_TOKEN" \
      "https://api.github.com/repos/$OWNER/$REPO/commits/$LATEST_SHA/status" | jq -r '.state')
    if [[ "$STATUS" == "success" ]]; then
      echo "✅ CI passed. Proceeding to deploy."
      break
    elif [[ "$STATUS" == "failure" ]]; then
      echo "❌ CI failed for $LATEST_SHA; aborting deploy." >&2
      exit 1
    else
      echo "🔄 CI status = $STATUS; retry in 30s …"
      sleep 30
    fi
  done
fi

echo "🚀 Running deployment script …"
if ./scripts/deploy-production.sh; then
  echo "✅ Deploy succeeded $(date)"
else
  echo "❌ Deploy failed, restoring previous PM2 snapshot" >&2
  pm2 resurrect || true
  exit 1
fi
