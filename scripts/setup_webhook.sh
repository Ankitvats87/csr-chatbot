#!/usr/bin/env bash
# Registers the Telegram webhook with the Bot API.
# Run AFTER `docker compose up -d` and AFTER Caddy has obtained the TLS cert.
#
# Reads .env from the project root.

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
  echo "ERROR: .env not found. Copy .env.example to .env and fill in values." >&2
  exit 1
fi

# shellcheck disable=SC1091
set -a; source .env; set +a

: "${TELEGRAM_BOT_TOKEN:?TELEGRAM_BOT_TOKEN is required in .env}"
: "${PUBLIC_DOMAIN:?PUBLIC_DOMAIN is required in .env}"
WEBHOOK_PATH="${WEBHOOK_PATH:-/webhook/telegram}"

# Auto-generate a webhook secret if missing.
if [[ -z "${TELEGRAM_WEBHOOK_SECRET:-}" || "$TELEGRAM_WEBHOOK_SECRET" == "REPLACE_ME" ]]; then
  NEW_SECRET=$(head -c 32 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 48)
  echo "Generated TELEGRAM_WEBHOOK_SECRET: $NEW_SECRET"
  if grep -q '^TELEGRAM_WEBHOOK_SECRET=' .env; then
    sed -i.bak "s|^TELEGRAM_WEBHOOK_SECRET=.*$|TELEGRAM_WEBHOOK_SECRET=${NEW_SECRET}|" .env
  else
    echo "TELEGRAM_WEBHOOK_SECRET=${NEW_SECRET}" >> .env
  fi
  TELEGRAM_WEBHOOK_SECRET="$NEW_SECRET"
  echo "Restart the app container so it picks up the new secret:  docker compose restart app"
fi

WEBHOOK_URL="https://${PUBLIC_DOMAIN}${WEBHOOK_PATH}"

echo "Setting Telegram webhook to: $WEBHOOK_URL"
curl -sS -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/setWebhook" \
  -H "Content-Type: application/json" \
  -d "$(cat <<JSON
{
  "url": "${WEBHOOK_URL}",
  "secret_token": "${TELEGRAM_WEBHOOK_SECRET}",
  "drop_pending_updates": true,
  "allowed_updates": ["message", "edited_message", "callback_query"]
}
JSON
)" | tee /tmp/setwebhook.json

echo
echo "Verifying:"
curl -sS "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getWebhookInfo" | tee /tmp/getwebhookinfo.json
echo
echo "Done. If 'ok': true and url matches, you're set."
