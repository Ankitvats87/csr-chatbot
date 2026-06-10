# Spark63 CSR Telegram RAG Bot

A production-ready Telegram bot that answers internal CSR questions using Retrieval-Augmented Generation (Pinecone + OpenAI embeddings + OpenAI chat), with automatic Google Drive ingestion and admin-managed access — all on a Hostinger VPS.

This README is a **step-by-step deployment runbook for `srv988340.hstgr.cloud`**. Follow it top to bottom on a fresh VPS and you will end up with a working bot.

---

## Features

- Telegram webhook bot (`@spark63bot`), HTTPS via Caddy + Let's Encrypt
- **RAG Optimization**: OpenAI embeddings (`text-embedding-3-small`) → Pinecone top-K=5 (threshold $\ge$ 0.50) → GPT-4o-mini → reply with source citations
- **Context-Enriched Ingestion**: Standalone utility (`scripts/ingest_local_file.py`) that parses compiled meeting transcripts, identifies meeting headers, and prepends them as context tags (`[Context: <Meeting Title>]\n\n<Chunk Text>`) to prevent orphaned chunks and maintain contextual integrity.
- **Web Admin Dashboard**: View connection health (SQLite, Pinecone, Drive, OpenAI), manage ingested files, and trigger manual Drive syncing with real-time UI status.
- Short conversational memory (last 10 turns per chat) in SQLite
- **Google Drive auto-ingestion**: polls a folder every 2 min; handles add, update, and delete; supports PDF/DOCX/TXT/CSV + Google Docs/Sheets
- **Admin-managed allowlist** with inline Approve/Deny buttons and Telegram commands (`/grant`, `/revoke`, `/list`, `/pending`)
- Docker Compose stack, `restart: unless-stopped` on every service
- Structured JSON logs, request/response tracking in SQLite (`data/sqlite/app.db`)

---

## Prerequisites

Before deploying, you need:

| What | How to get it |
|---|---|
| Telegram Bot Token | DM `@BotFather` on Telegram → `/mybots` → pick your bot → **API Token** |
| Pinecone API key | https://app.pinecone.io → API keys. The bot auto-creates the index if missing. |
| OpenAI API key | https://platform.openai.com/api-keys — needed for embeddings and chat responses |
| Google Drive OAuth credentials | See "Generating Google Drive credentials" below |
| Your Telegram chat_id | Optional — the first user to send `/start` is auto-claimed as admin |

---

## Step 1 — Connect to the VPS

```bash
ssh root@srv988340.hstgr.cloud
```

## Step 2 — Install Docker (one-time)

```bash
curl -fsSL https://get.docker.com | sh
systemctl enable --now docker
docker compose version   # should print v2.x
```

## Step 3 — Open firewall ports

```bash
ufw allow OpenSSH
ufw allow 80
ufw allow 443
ufw --force enable
ufw status
```

## Step 4 — Copy the code onto the VPS

```bash
# From your laptop (Windows PowerShell):
scp -r "G:/My Drive/AI/Projects/CSR Bot/CSR Chatbot/telegram-rag-bot" root@srv988340.hstgr.cloud:/opt/csr-bot
ssh root@srv988340.hstgr.cloud
cd /opt/csr-bot
chmod +x scripts/*.sh
```

## Step 5 — Configure `.env`

```bash
cp .env.example .env
nano .env
```

Fill in **at minimum**:

```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=spark-csr
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
PUBLIC_DOMAIN=srv988340.hstgr.cloud

# Web Admin Dashboard Authentication
ADMIN_USERNAME=admin
ADMIN_PASSWORD=strongpassword

# Google Drive — see credentials section below
GOOGLE_DRIVE_CLIENT_ID=...
GOOGLE_DRIVE_CLIENT_SECRET=...
GOOGLE_DRIVE_REFRESH_TOKEN=...
GOOGLE_DRIVE_FOLDER_ID=...

# LlamaParse (LlamaIndex Cloud) layout-aware parser key
LLAMA_CLOUD_API_KEY=llx-...
```

You can leave `TELEGRAM_WEBHOOK_SECRET` blank — `setup_webhook.sh` will generate and persist one for you. You can leave `ADMIN_CHAT_IDS` blank too — the first user to send `/start` auto-claims admin.

---

## Localhost Development & Testing

To test and review the bot entirely on your local machine without needing a VPS or public HTTPS:

1. **Start the Local Server**:
   ```bash
   pip install -r requirements.txt
   uvicorn app.main:app --reload --port 8000
   ```
   *The FastAPI backend is now running at `http://localhost:8000`.*

2. **Launch the Local Polling Bridge**:
   In a separate terminal, run:
   ```bash
   python scripts/local_poll.py
   ```
   This script deletes any remote webhook configuration and relays messages from Telegram directly to your localhost web server, passing the required authorization headers.

3. **Access the Admin Dashboard**:
   Open a browser and go to `http://localhost:8000/admin`.
   Log in with the credentials set in your `.env` (`ADMIN_USERNAME` and `ADMIN_PASSWORD`).

---

## Hostinger VPS Deployment

To host this application in production on your Hostinger VPS (`srv988340.hstgr.cloud`):

### Step 1 — SSH into VPS
```bash
ssh root@srv988340.hstgr.cloud
```

### Step 2 — Install Docker
```bash
curl -fsSL https://get.docker.com | sh
systemctl enable --now docker
```

### Step 3 — Open firewall ports
```bash
ufw allow OpenSSH
ufw allow 80
ufw allow 443
ufw --force enable
```

### Step 4 — Upload Code to VPS
From your laptop, copy the project files to `/opt/csr-bot`:
```bash
scp -r "G:/My Drive/AI/Projects/CSR Bot/CSR Chatbot/telegram-rag-bot" root@srv988340.hstgr.cloud:/opt/csr-bot
```
Then SSH in and make the shell scripts executable:
```bash
ssh root@srv988340.hstgr.cloud
cd /opt/csr-bot
chmod +x scripts/*.sh
```

### Step 5 — Deploy
```bash
bash scripts/deploy.sh
```
What this does:
1. Builds the Docker image.
2. Launches the backend (`csrbot-app`), the ingestion pipeline (`csrbot-ingestion`), and the Caddy web server (`csrbot-caddy`).
3. Provisions a free SSL certificate via Let's Encrypt using ACME.
4. Registers your Telegram bot webhook pointing to `https://srv988340.hstgr.cloud/webhook/telegram`.

### Step 6 — Access the Dashboard Remotely
Go to `https://srv988340.hstgr.cloud/admin` and log in with your dashboard credentials.

---

## Generating Google Drive credentials

You need a Client ID, Client Secret, and Refresh Token from a one-time OAuth flow.

### Option A — Quickest: use Google OAuth Playground

1. Create credentials in Google Cloud:
   - Go to https://console.cloud.google.com → create a project (or pick existing).
   - **APIs & Services → Library** → enable **Google Drive API**.
   - **APIs & Services → OAuth consent screen** → External → fill required fields → add yourself as a Test user.
   - **APIs & Services → Credentials → Create Credentials → OAuth client ID** → Application type: **Web application** → Authorized redirect URIs: `https://developers.google.com/oauthplayground` → Create. Copy the **Client ID** and **Client secret**.

2. Generate the refresh token:
   - Open https://developers.google.com/oauthplayground
   - Click the gear (⚙) top-right → check **Use your own OAuth credentials** → paste your Client ID + Secret.
   - In the left panel, find **Drive API v3** and select scope: `https://www.googleapis.com/auth/drive.readonly`
   - Click **Authorize APIs** → sign in with the Google account that owns the Drive folder → grant.
   - Click **Exchange authorization code for tokens** → copy the **Refresh token**.

3. Get the folder ID:
   - Open your Drive folder in a browser. The URL looks like `https://drive.google.com/drive/folders/XXXXXXXXXX` — that final segment is the `GOOGLE_DRIVE_FOLDER_ID`.

4. Paste all four values into `.env`.

---

## Ingesting Local Files with Context Enrichment

If you have a large compiled text document (like `Test CSR 21_26 .txt` in `data/raw_docs/`) containing multiple meeting notes separated by delimiters (like `==================================================`), you can ingest it locally to Pinecone with context tags:

```bash
python scripts/ingest_local_file.py "data/raw_docs/Test CSR 21_26 .txt"
```

### Critical Operations Warnings

1. **Conflict with Google Drive Sync**:
   The Admin Dashboard "Sync" button (or Caddy's auto-sync) runs a comparison against the Google Drive folder. Since local files are not in Google Drive, **syncing will automatically delete local index files and vectors from Pinecone**. For permanent indexing, upload files to Google Drive.
   
2. **Pinecone Asynchronous Deletion Race Condition**:
   Pinecone's metadata delete-by-filter API is eventually consistent. When re-indexing a file, the system sleeps for 5 seconds (`time.sleep(5)`) after issuing the delete command. This ensures Pinecone deletes the old vectors before the new batches are uploaded, preventing the new vectors from being accidentally deleted.

---

## Admin onboarding (bootstrap + invite flow)

### Bootstrap
- If `ADMIN_CHAT_IDS` is blank in `.env`, the **first `/start`** message auto-claims that user as admin.
- After it happens, the bot tells you your chat_id — paste it into `.env` as `ADMIN_CHAT_IDS=...` so it survives container restarts, then `docker compose restart app`.

### Inviting a new user
1. Tell the new user to open your bot and press **Start**.
2. You (admin) get a Telegram message:
   ```
   Access request from @newuser
   chat_id: 12345678
   Approve or deny: [Approve] [Deny]
   ```
3. Tap **Approve** — they get a "You're in" DM instantly.

### Admin commands

| Command | What it does |
|---|---|
| `/grant <chat_id\|@username>` | Allow a user |
| `/revoke <chat_id\|@username>` | Remove a user immediately |
| `/list` | Show all active users |
| `/pending` | Show users waiting for approval |
| `/ingest` | Show ingestion sync commands |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/health` returns pinecone:false | Check `PINECONE_API_KEY` + `PINECONE_INDEX_NAME` in `.env`, then `docker compose restart app` |
| Webhook returns 403 | Secret mismatch; re-run `bash scripts/setup_webhook.sh` and `docker compose restart app` |
| Caddy fails to get a cert | Confirm ports 80 + 443 are open and DNS resolves to this VPS, and `ACME_EMAIL` is a real address |
| "You are not authorised" | Send `/start` first → admin approves. Or add your chat_id to `ADMIN_CHAT_IDS` in `.env` and restart |
| Replies are fallback "I don't have that information" | Cosine similarity scores for relevant chunks are often low (0.52-0.58). Lower `SIMILARITY_THRESHOLD=0.50` in `.env` and restart uvicorn |

---

## Architecture

```
Telegram → Caddy (TLS) → FastAPI /webhook/telegram (csrbot-app)
                              │
                              ▼
                      ┌── RAGService
                      │   ├── EmbeddingService (OpenAI)
                      │   ├── VectorService (Pinecone, top-K=5, ≥0.50)
                      │   ├── MemoryService (SQLite, last 10 turns)
                      │   ├── PromptService
                      │   └── ResponseService (OpenAI gpt-4o-mini)
                      │
                      ├── AccessService (allowlist, admin commands, inline approval)
                      └── TelegramService (Bot API)

                        (separate container: csrbot-ingestion)
                        APScheduler every 2 min
                              │
                              ▼
                        IngestionService
                        ├── DriveLoader (list + download)
                        ├── Chunker (PDF/DOCX/TXT/CSV → segments)
                        ├── embed_and_upsert (batched + 5s sleep)
                        └── delete_by_file_id on disappearance
```
