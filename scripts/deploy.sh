#!/usr/bin/env bash
# One-shot deploy on a Linux VPS (Hostinger srv988340.hstgr.cloud).
#
# Prereqs (one-time):
#   - Docker + Docker Compose plugin installed
#   - .env populated (copy .env.example -> .env, fill in keys)
#   - Ports 80 + 443 open in firewall, DNS for PUBLIC_DOMAIN points at this VPS
#
# Idempotent: safe to re-run after pulling new code.

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  echo "ERROR: .env not found. Run: cp .env.example .env  and fill in values." >&2
  exit 1
fi

# Required values that must not be the placeholder.
for var in TELEGRAM_BOT_TOKEN PINECONE_API_KEY PINECONE_INDEX_NAME PUBLIC_DOMAIN; do
  val=$(grep -E "^${var}=" .env | head -n1 | cut -d= -f2- || true)
  if [[ -z "$val" || "$val" == "REPLACE_ME" ]]; then
    echo "ERROR: $var is missing or REPLACE_ME in .env" >&2
    exit 1
  fi
done

echo "==> Building app image"
docker compose build app

echo "==> Starting stack"
docker compose up -d

echo "==> Waiting for app health (up to 60s)"
for i in $(seq 1 30); do
  if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

if ! curl -fsS http://localhost:8000/health; then
  echo
  echo "ERROR: app did not become healthy. Logs:" >&2
  docker compose logs --tail=80 app
  exit 1
fi
echo

echo "==> Registering Telegram webhook"
bash scripts/setup_webhook.sh

echo
echo "Deploy complete."
echo "Tail logs with:  docker compose logs -f app"
