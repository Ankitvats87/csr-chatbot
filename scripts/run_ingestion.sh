#!/usr/bin/env bash
# Runs a single Google Drive -> Pinecone sync pass inside the ingestion container.
# Use this to force an immediate scan instead of waiting for the next poll.

set -euo pipefail

cd "$(dirname "$0")/.."

# Prefer running inside the existing ingestion container.
if docker compose ps ingestion --status running --quiet >/dev/null 2>&1; then
  exec docker compose exec ingestion python -m app.ingestion.scheduler --once
fi

# Fallback: spin up a one-shot container.
exec docker compose run --rm ingestion python -m app.ingestion.scheduler --once
