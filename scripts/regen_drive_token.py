"""Regenerate Google Drive OAuth refresh token.

Runs a one-shot loopback OAuth flow on http://localhost:8765.
- Prints the Google consent URL.
- Spins up a tiny HTTP server to catch the redirect.
- Exchanges the authorization code for a refresh token.
- Updates .env in-place (creates a .env.bak first).

Usage:
    cd telegram-rag-bot
    python -u scripts/regen_drive_token.py

Prerequisite — your OAuth client in Google Cloud Console must have
`http://localhost:8765` listed under "Authorized redirect URIs".
If it is a Desktop-app type client, localhost is implicitly allowed.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import secrets
import socketserver
import sys
import threading
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"

PORT = 8765
# Must EXACTLY match an Authorized redirect URI in your Google Cloud Console OAuth client.
REDIRECT_PATH = "/"
REDIRECT_URI = f"http://localhost:{PORT}{REDIRECT_PATH}"
SCOPE = "https://www.googleapis.com/auth/drive.readonly"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"


# ── Capture the authorization code from the redirect ────────────────────────

class _CodeReceiver:
    code: str | None = None
    state_seen: str | None = None
    error: str | None = None


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        _CodeReceiver.code = (params.get("code") or [None])[0]
        _CodeReceiver.state_seen = (params.get("state") or [None])[0]
        _CodeReceiver.error = (params.get("error") or [None])[0]

        body = (
            "<html><body style='font-family:sans-serif;padding:40px'>"
            "<h2>Authorization received.</h2>"
            "<p>You can close this tab and return to the terminal.</p>"
            "</body></html>"
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, *args, **kwargs):  # silence
        return


def _start_server() -> socketserver.TCPServer:
    server = socketserver.TCPServer(("127.0.0.1", PORT), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


# ── Update .env in place ────────────────────────────────────────────────────

def _update_env(new_token: str) -> None:
    bak = ENV_PATH.with_suffix(".env.bak")
    bak.write_bytes(ENV_PATH.read_bytes())

    lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    found = False
    out: list[str] = []
    for ln in lines:
        if ln.startswith("GOOGLE_DRIVE_REFRESH_TOKEN="):
            out.append(f"GOOGLE_DRIVE_REFRESH_TOKEN={new_token}")
            found = True
        else:
            out.append(ln)
    if not found:
        out.append(f"GOOGLE_DRIVE_REFRESH_TOKEN={new_token}")

    ENV_PATH.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"[OK] Wrote new refresh token to {ENV_PATH}")
    print(f"[OK] Backup saved to {bak}")


# ── Main flow ───────────────────────────────────────────────────────────────

def main() -> int:
    global PORT, REDIRECT_URI
    parser = argparse.ArgumentParser(description="Regenerate Google Drive OAuth refresh token.")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run loopback server on (default: 8765, or loads APP_PORT from .env)",
    )
    args = parser.parse_args()

    load_dotenv(ENV_PATH)

    # Determine port: CLI argument -> .env APP_PORT -> fallback 8765
    if args.port is not None:
        PORT = args.port
    else:
        env_port = os.getenv("APP_PORT")
        if env_port:
            try:
                PORT = int(env_port)
            except ValueError:
                PORT = 8765
        else:
            PORT = 8765

    REDIRECT_URI = f"http://localhost:{PORT}{REDIRECT_PATH}"

    client_id = os.getenv("GOOGLE_DRIVE_CLIENT_ID", "").strip()
    client_secret = os.getenv("GOOGLE_DRIVE_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        print("ERROR: GOOGLE_DRIVE_CLIENT_ID / GOOGLE_DRIVE_CLIENT_SECRET missing in .env")
        return 1

    state = secrets.token_urlsafe(16)
    params = {
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": SCOPE,
        "access_type": "offline",      # required for refresh_token
        "prompt": "consent",            # force re-consent so a NEW refresh_token is issued
        "state": state,
    }
    auth_url = f"{AUTH_URL}?{urllib.parse.urlencode(params)}"

    print("=" * 70)
    print("  STEP 1.  Open this URL in your browser, sign in, click Allow:")
    print("=" * 70)
    print()
    print(auth_url)
    print()
    print("=" * 70)
    print(f"  STEP 2.  Waiting for Google to redirect to {REDIRECT_URI} ...")
    print("=" * 70)

    server = _start_server()
    try:
        webbrowser.open(auth_url)
    except Exception:
        pass  # not critical; user can copy-paste

    # Wait up to 15 minutes for the redirect
    import time
    deadline = time.time() + 900
    while time.time() < deadline:
        if _CodeReceiver.code or _CodeReceiver.error:
            break
        time.sleep(0.5)
    server.shutdown()

    if _CodeReceiver.error:
        print(f"\nERROR from Google: {_CodeReceiver.error}")
        return 1
    if not _CodeReceiver.code:
        print("\nERROR: timed out waiting for redirect. Did you click Allow?")
        return 1
    if _CodeReceiver.state_seen != state:
        print("\nERROR: state mismatch. Aborting.")
        return 1

    print("\n[OK] Authorization code captured. Exchanging for refresh token...")

    body = urllib.parse.urlencode({
        "code": _CodeReceiver.code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    }).encode()

    req = urllib.request.Request(
        TOKEN_URL,
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode(errors="ignore")
        print(f"\nERROR exchanging code: HTTP {e.code}\n{err_body}")
        return 1

    refresh_token = payload.get("refresh_token")
    if not refresh_token:
        print("\nERROR: No refresh_token in response. Google may have returned only an access_token.")
        print(f"Response: {payload}")
        print("\nFix: revoke prior consent at https://myaccount.google.com/permissions, then retry.")
        return 1

    _update_env(refresh_token)
    print()
    print("Done. Restart FastAPI for the new token to take effect:")
    print("  1. Stop the current uvicorn process")
    print("  2. cd telegram-rag-bot && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
