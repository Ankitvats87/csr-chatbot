import asyncio
import hashlib
import secrets
from typing import Dict, List
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()
security = HTTPBasic()


def get_current_admin(credentials: HTTPBasicCredentials = Depends(security), request: Request = None):
    # Fetch dependencies from app state
    deps = request.app.state.deps
    sqlite = deps.sqlite
    settings = deps.settings

    # Try loading custom credentials from DB
    db_username = None
    db_password_hash = None

    try:
        u_row = sqlite.fetchone("SELECT value FROM web_admin_config WHERE key = 'username'")
        p_row = sqlite.fetchone("SELECT value FROM web_admin_config WHERE key = 'password_hash'")
        if u_row:
            db_username = u_row["value"]
        if p_row:
            db_password_hash = p_row["value"]
    except Exception as e:
        logger.warning(f"Failed to read admin credentials from SQLite: {e}")

    # Authentication check
    correct_username = False
    correct_password = False

    if db_username and db_password_hash:
        # DB credentials are set. Use SHA256 comparison.
        correct_username = secrets.compare_digest(credentials.username, db_username)
        input_hash = hashlib.sha256(credentials.password.encode("utf-8")).hexdigest()
        correct_password = secrets.compare_digest(input_hash, db_password_hash)
    else:
        # Fallback to .env settings (defaults: admin/admin)
        env_user = settings.admin_username or "admin"
        env_pass = settings.admin_password or "admin"
        correct_username = secrets.compare_digest(credentials.username, env_user)
        correct_password = secrets.compare_digest(credentials.password, env_pass)

    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


# HTML Dashboard
@router.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request, username: str = Depends(get_current_admin)):
    deps = request.app.state.deps
    folder_id = deps.settings.google_drive_folder_id or "Not Configured"
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Spark63 CSR Bot - Admin Dashboard</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg-color: #0b0f19;
                --card-bg: rgba(22, 28, 45, 0.45);
                --card-border: rgba(255, 255, 255, 0.08);
                --text-main: #f3f4f6;
                --text-muted: #9ca3af;
                --primary: #4f46e5;
                --primary-hover: #4338ca;
                --accent-emerald: #10b981;
                --accent-rose: #f43f5e;
                --accent-amber: #f59e0b;
                --shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            }}

            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }}

            body {{
                font-family: 'Outfit', sans-serif;
                background-color: var(--bg-color);
                color: var(--text-main);
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                background-image: radial-gradient(circle at 10% 20%, rgba(99, 102, 241, 0.15) 0%, transparent 40%),
                                  radial-gradient(circle at 90% 80%, rgba(16, 185, 129, 0.1) 0%, transparent 40%);
                background-attachment: fixed;
            }}

            header {{
                background: rgba(11, 15, 25, 0.8);
                backdrop-filter: blur(12px);
                border-bottom: 1px solid var(--card-border);
                padding: 1.25rem 2rem;
                position: sticky;
                top: 0;
                z-index: 100;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}

            header h1 {{
                font-size: 1.5rem;
                font-weight: 700;
                background: linear-gradient(135deg, #a5b4fc 0%, #818cf8 50%, #4f46e5 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}

            .user-tag {{
                font-size: 0.875rem;
                color: var(--text-muted);
                background: var(--card-bg);
                border: 1px solid var(--card-border);
                padding: 0.4rem 0.8rem;
                border-radius: 50px;
            }}

            main {{
                flex-grow: 1;
                max-width: 1200px;
                width: 100%;
                margin: 2rem auto;
                padding: 0 1.5rem;
                display: flex;
                flex-direction: column;
                gap: 2rem;
            }}

            .grid-stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 1.5rem;
            }}

            .stat-card {{
                background: var(--card-bg);
                border: 1px solid var(--card-border);
                border-radius: 16px;
                padding: 1.5rem;
                backdrop-filter: blur(8px);
                box-shadow: var(--shadow);
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
                transition: transform 0.2s, border-color 0.2s;
            }}

            .stat-card:hover {{
                transform: translateY(-2px);
                border-color: rgba(99, 102, 241, 0.3);
            }}

            .stat-label {{
                font-size: 0.875rem;
                color: var(--text-muted);
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }}

            .stat-value {{
                font-size: 1.5rem;
                font-weight: 700;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }}

            .status-indicator {{
                width: 10px;
                height: 10px;
                border-radius: 50%;
                display: inline-block;
            }}

            .status-indicator.ok {{
                background-color: var(--accent-emerald);
                box-shadow: 0 0 10px var(--accent-emerald);
            }}

            .status-indicator.error {{
                background-color: var(--accent-rose);
                box-shadow: 0 0 10px var(--accent-rose);
            }}

            .status-indicator.unknown {{
                background-color: var(--accent-amber);
                box-shadow: 0 0 10px var(--accent-amber);
            }}

            .dashboard-sections {{
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 1.5rem;
            }}

            @media(max-width: 900px) {{
                .dashboard-sections {{
                    grid-template-columns: 1fr;
                }}
            }}

            .card {{
                background: var(--card-bg);
                border: 1px solid var(--card-border);
                border-radius: 16px;
                padding: 2rem;
                backdrop-filter: blur(8px);
                box-shadow: var(--shadow);
                display: flex;
                flex-direction: column;
                gap: 1.5rem;
            }}

            .card-title {{
                font-size: 1.25rem;
                font-weight: 600;
                border-bottom: 1px solid var(--card-border);
                padding-bottom: 0.75rem;
            }}

            .instruction-box {{
                background: rgba(79, 70, 229, 0.1);
                border-left: 4px solid var(--primary);
                padding: 1rem;
                border-radius: 0 8px 8px 0;
                font-size: 0.925rem;
                line-height: 1.5;
            }}

            .instruction-box code {{
                background: rgba(0, 0, 0, 0.3);
                padding: 0.2rem 0.4rem;
                border-radius: 4px;
                color: #a5b4fc;
                font-family: monospace;
            }}

            .btn {{
                background: var(--primary);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0.75rem 1.5rem;
                font-size: 0.95rem;
                font-weight: 600;
                cursor: pointer;
                transition: background-color 0.2s, transform 0.1s;
                display: flex;
                justify-content: center;
                align-items: center;
                gap: 0.5rem;
            }}

            .btn:hover {{
                background: var(--primary-hover);
            }}

            .btn:active {{
                transform: scale(0.98);
            }}

            .btn:disabled {{
                background: #4b5563;
                cursor: not-allowed;
            }}

            .files-table-container {{
                overflow-x: auto;
                border-radius: 8px;
                border: 1px solid var(--card-border);
            }}

            .files-table {{
                width: 100%;
                border-collapse: collapse;
                text-align: left;
                font-size: 0.9rem;
            }}

            .files-table th {{
                background: rgba(0, 0, 0, 0.2);
                padding: 0.75rem 1rem;
                font-weight: 600;
                border-bottom: 1px solid var(--card-border);
                color: var(--text-muted);
            }}

            .files-table td {{
                padding: 0.75rem 1rem;
                border-bottom: 1px solid var(--card-border);
            }}

            .files-table tr:last-child td {{
                border-bottom: none;
            }}

            .badge {{
                display: inline-block;
                padding: 0.2rem 0.5rem;
                font-size: 0.75rem;
                font-weight: 600;
                border-radius: 4px;
                text-transform: uppercase;
            }}

            .badge.indexed {{
                background: rgba(16, 185, 129, 0.15);
                color: var(--accent-emerald);
            }}

            .badge.pending {{
                background: rgba(245, 158, 11, 0.15);
                color: var(--accent-amber);
            }}

            .badge.failed {{
                background: rgba(244, 63, 94, 0.15);
                color: var(--accent-rose);
            }}

            .form-group {{
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
                margin-bottom: 1.25rem;
            }}

            .form-group label {{
                font-size: 0.875rem;
                color: var(--text-muted);
                font-weight: 600;
            }}

            .form-control {{
                background: rgba(0, 0, 0, 0.3);
                border: 1px solid var(--card-border);
                border-radius: 8px;
                padding: 0.75rem;
                color: var(--text-main);
                font-family: inherit;
                font-size: 0.95rem;
            }}

            .form-control:focus {{
                border-color: var(--primary);
                outline: none;
            }}

            footer {{
                text-align: center;
                padding: 2rem;
                color: var(--text-muted);
                font-size: 0.85rem;
                border-top: 1px solid var(--card-border);
                background: rgba(11, 15, 25, 0.5);
            }}

            /* Animations & micro-interactions */
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}

            .spinner {{
                width: 18px;
                height: 18px;
                border: 2px solid rgba(255, 255, 255, 0.3);
                border-radius: 50%;
                border-top-color: white;
                animation: spin 0.8s linear infinite;
                display: none;
            }}

            .syncing .spinner {{
                display: inline-block;
            }}

            .sync-result-box {{
                display: none;
                margin-top: 1rem;
                padding: 1rem;
                border-radius: 8px;
                font-size: 0.875rem;
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid var(--card-border);
                line-height: 1.4;
            }}
        </style>
    </head>
    <body>
        <header>
            <h1>Spark63 CSR Bot Administration</h1>
            <div class="user-tag">Admin: {username}</div>
        </header>

        <main>
            <div class="grid-stats">
                <div class="stat-card">
                    <div class="stat-label">FastAPI Backend</div>
                    <div class="stat-value">
                        <span class="status-indicator ok"></span>
                        Active
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">SQLite Client</div>
                    <div class="stat-value" id="status-sqlite">
                        <span class="status-indicator unknown"></span>
                        Checking...
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Pinecone Index</div>
                    <div class="stat-value" id="status-pinecone">
                        <span class="status-indicator unknown"></span>
                        Checking...
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Google Drive Config</div>
                    <div class="stat-value" id="status-drive">
                        <span class="status-indicator unknown"></span>
                        Checking...
                    </div>
                </div>
            </div>

            <div class="dashboard-sections">
                <!-- Left panel: File Management & Ingestion -->
                <div class="card">
                    <div class="card-title">Knowledge Base Ingestion</div>
                    
                    <div class="instruction-box">
                        <strong>Pre-ingestion instructions:</strong><br>
                        Please make sure you upload or update your files (PDF, DOCX, TXT, CSV) inside your configured Google Drive folder.
                        <br>
                        Current Google Drive Folder ID: <code>{folder_id}</code>
                    </div>

                    <div style="display: flex; flex-direction: column; gap: 1rem;">
                        <div style="display: flex; gap: 1rem;">
                            <button id="btn-sync" class="btn" style="flex: 2;" onclick="triggerSync()">
                                <div class="spinner"></div>
                                <span id="btn-text">Sync Google Drive Folder</span>
                            </button>
                            <a href="/admin/google-auth-init" class="btn" style="flex: 1; text-decoration: none; background-color: var(--accent-amber); text-align: center; display: flex; justify-content: center; align-items: center;">
                                Authorize Drive
                            </a>
                        </div>
                        <div id="sync-result" class="sync-result-box"></div>
                    </div>

                    <div class="card-title" style="margin-top: 1rem; border-bottom: 1px solid var(--card-border); padding-bottom: 0.5rem;">Ingested Files ({deps.settings.pinecone_namespace} namespace)</div>
                    <div class="files-table-container">
                        <table class="files-table">
                            <thead>
                                <tr>
                                    <th>File Name</th>
                                    <th>Modified Time</th>
                                    <th>Status</th>
                                    <th>Chunks</th>
                                    <th>Last Indexed</th>
                                </tr>
                            </thead>
                            <tbody id="files-list">
                                <tr>
                                    <td colspan="5" style="text-align: center; color: var(--text-muted); padding: 2rem;">Loading files list...</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- Right panel: Credentials / Password settings -->
                <div class="card" style="height: fit-content;">
                    <div class="card-title">Change Admin Password</div>
                    <form id="pw-form" onsubmit="changePassword(event)">
                        <div class="form-group">
                            <label for="new-username">Username</label>
                            <input type="text" id="new-username" class="form-control" placeholder="New username" required value="{username}">
                        </div>
                        <div class="form-group">
                            <label for="new-password">New Password</label>
                            <input type="password" id="new-password" class="form-control" placeholder="New password" required>
                        </div>
                        <div class="form-group">
                            <label for="confirm-password">Confirm Password</label>
                            <input type="password" id="confirm-password" class="form-control" placeholder="Confirm password" required>
                        </div>
                        <button type="submit" class="btn" style="width: 100%;">Update Credentials</button>
                    </form>
                    <div id="pw-result" style="display: none; padding: 0.75rem; border-radius: 8px; font-size: 0.875rem; margin-top: 0.5rem; text-align: center;"></div>
                </div>
            </div>
        </main>

        <footer>
            Spark63 CSR RAG Bot Service v1.0.0 &copy; 2026. Made with Vanilla CSS.
        </footer>

        <script>
            // Populate status and load file lists on boot
            window.addEventListener('DOMContentLoaded', () => {{
                checkStatus();
                loadFiles();
            }});

            async function checkStatus() {{
                try {{
                    const r = await fetch('/health');
                    const d = await r.json();

                    updateStatusIndicator('status-sqlite', d.sqlite);
                    updateStatusIndicator('status-pinecone', d.pinecone);
                    
                    // Drive configuration check
                    const driveConfigured = d.llm_provider !== undefined; // Simple placeholder check
                    updateStatusIndicator('status-drive', driveConfigured);
                }} catch (e) {{
                    console.error("Health check failed", e);
                }}
            }}

            function updateStatusIndicator(elementId, isOk) {{
                const el = document.getElementById(elementId);
                if (el) {{
                    el.innerHTML = isOk 
                        ? '<span class="status-indicator ok"></span> Connected' 
                        : '<span class="status-indicator error"></span> Connection Failed';
                }}
            }}

            async function loadFiles() {{
                try {{
                    const r = await fetch('/admin/files');
                    const files = await r.json();
                    const tbody = document.getElementById('files-list');
                    
                    if (files.length === 0) {{
                        tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted); padding: 2rem;">No files ingested yet. Click Sync to begin.</td></tr>';
                        return;
                    }}

                    tbody.innerHTML = files.map(f => `
                        <tr>
                            <td style="font-weight: 600;">${{f.name}}</td>
                            <td>${{new Date(f.modified_time).toLocaleString()}}</td>
                            <td><span class="badge ${{f.status}}">${{f.status}}</span></td>
                            <td style="font-family: monospace;">${{f.n_chunks}}</td>
                            <td>${{f.last_indexed ? new Date(f.last_indexed).toLocaleString() : 'N/A'}}</td>
                        </tr>
                    `).join('');
                }} catch (e) {{
                    console.error("Failed to load files", e);
                }}
            }}

            async function triggerSync() {{
                const btn = document.getElementById('btn-sync');
                const btnText = document.getElementById('btn-text');
                const resultBox = document.getElementById('sync-result');

                btn.disabled = true;
                btn.classList.add('syncing');
                btnText.textContent = "Syncing Drive Folder...";
                resultBox.style.display = 'none';

                try {{
                    const r = await fetch('/admin/sync', {{ method: 'POST' }});
                    const d = await r.json();

                    resultBox.style.display = 'block';
                    if (d.status === 'ok') {{
                        resultBox.style.color = '#34d399';
                        resultBox.innerHTML = `
                            <strong>Drive sync completed successfully!</strong><br>
                            &bull; Added: ${{d.results.added}}<br>
                            &bull; Updated: ${{d.results.updated}}<br>
                            &bull; Deleted: ${{d.results.deleted}}<br>
                            &bull; Failed: ${{d.results.failed}}
                        `;
                    }} else {{
                        resultBox.style.color = '#f87171';
                        resultBox.innerHTML = `<strong>Sync warning:</strong> ${{d.message || 'Details in backend logs'}}`;
                    }}

                    // Reload files list and connection state
                    await loadFiles();
                    await checkStatus();
                }} catch (e) {{
                    resultBox.style.display = 'block';
                    resultBox.style.color = '#f87171';
                    resultBox.innerHTML = `<strong>Error triggering sync:</strong> Connection error`;
                }} finally {{
                    btn.disabled = false;
                    btn.classList.remove('syncing');
                    btnText.textContent = "Sync Google Drive Folder";
                }}
            }}

            async function changePassword(e) {{
                e.preventDefault();
                const user = document.getElementById('new-username').value;
                const pass = document.getElementById('new-password').value;
                const confirm = document.getElementById('confirm-password').value;
                const resBox = document.getElementById('pw-result');

                resBox.style.display = 'none';

                if (pass !== confirm) {{
                    resBox.style.display = 'block';
                    resBox.style.backgroundColor = 'rgba(244, 63, 94, 0.15)';
                    resBox.style.color = 'var(--accent-rose)';
                    resBox.textContent = 'Passwords do not match.';
                    return;
                }}

                try {{
                    const r = await fetch('/admin/change-password', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ username: user, password: pass }})
                    }});
                    const d = await r.json();

                    resBox.style.display = 'block';
                    if (d.ok) {{
                        resBox.style.backgroundColor = 'rgba(16, 185, 129, 0.15)';
                        resBox.style.color = 'var(--accent-emerald)';
                        resBox.textContent = 'Credentials updated. Please use the new credentials next time.';
                        document.getElementById('pw-form').reset();
                        document.getElementById('new-username').value = user;
                    }} else {{
                        resBox.style.backgroundColor = 'rgba(244, 63, 94, 0.15)';
                        resBox.style.color = 'var(--accent-rose)';
                        resBox.textContent = d.message || 'Failed to update credentials.';
                    }}
                }} catch (err) {{
                    resBox.style.display = 'block';
                    resBox.style.backgroundColor = 'rgba(244, 63, 94, 0.15)';
                    resBox.style.color = 'var(--accent-rose)';
                    resBox.textContent = 'Failed to connect to server.';
                }}
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@router.get("/admin/files")
async def get_files(request: Request, username: str = Depends(get_current_admin)):
    deps = request.app.state.deps
    sqlite = deps.sqlite

    try:
        # Load all file indexing summaries from DB
        rows = sqlite.fetchall(
            "SELECT name, modified_time, last_indexed, n_chunks, status FROM ingested_files ORDER BY last_indexed DESC"
        )
        files = []
        for r in rows:
            files.append({
                "name": r["name"],
                "modified_time": r["modified_time"],
                "last_indexed": r["last_indexed"],
                "n_chunks": r["n_chunks"],
                "status": r["status"]
            })
        return JSONResponse(content=files)
    except Exception as e:
        logger.exception("Failed to query files table", extra={"err": str(e)})
        return JSONResponse(content=[], status_code=500)


@router.post("/admin/sync")
async def sync_now(request: Request, username: str = Depends(get_current_admin)):
    deps = request.app.state.deps
    if not deps.settings.drive_configured():
        return JSONResponse(status_code=400, content={"status": "error", "message": "Google Drive is not configured."})

    try:
        # Execute Drive sync synchronously in this thread worker (since FastAPI handles this route)
        # Wrapping it in asyncio.to_thread to keep route non-blocking.
        counts = await asyncio.to_thread(deps.ingestion.sync_once)
        if "skipped" in counts:
            return JSONResponse(status_code=500, content={"status": "error", "message": "Drive loader is not configured."})
        return JSONResponse(content={"status": "ok", "results": counts})
    except Exception as e:
        logger.exception("Dashboard sync trigger failed", extra={"err": str(e)})
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


from pydantic import BaseModel

class CredentialUpdate(BaseModel):
    username: str
    password: str

@router.post("/admin/change-password")
async def change_password(
    data: CredentialUpdate,
    request: Request,
    username: str = Depends(get_current_admin)
):
    deps = request.app.state.deps
    sqlite = deps.sqlite

    # Clean username
    clean_username = data.username.strip()
    if not clean_username or len(data.password) < 4:
        return JSONResponse(status_code=400, content={"ok": False, "message": "Username must not be empty. Password must be at least 4 characters."})

    # Hash password with SHA-256
    hashed_pass = hashlib.sha256(data.password.encode("utf-8")).hexdigest()

    try:
        sqlite.execute(
            "INSERT OR REPLACE INTO web_admin_config (key, value) VALUES ('username', ?)",
            (clean_username,)
        )
        sqlite.execute(
            "INSERT OR REPLACE INTO web_admin_config (key, value) VALUES ('password_hash', ?)",
            (hashed_pass,)
        )
        logger.info(f"Admin credentials updated in database by user {username}")
        return JSONResponse(content={"ok": True})
    except Exception as e:
        logger.exception("Failed to write new credentials to SQLite config", extra={"err": str(e)})
        return JSONResponse(status_code=500, content={"ok": False, "message": "Database write error."})


from fastapi.responses import RedirectResponse

@router.get("/admin/google-auth-init")
async def google_auth_init(request: Request, username: str = Depends(get_current_admin)):
    deps = request.app.state.deps
    settings = deps.settings
    client_id = settings.google_drive_client_id
    if not client_id:
        raise HTTPException(status_code=400, detail="GOOGLE_DRIVE_CLIENT_ID is not configured in .env")

    import urllib.parse
    import secrets

    state = secrets.token_urlsafe(16)
    redirect_uri = str(request.base_url).rstrip("/") + "/oauth2callback"

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "https://www.googleapis.com/auth/drive.readonly",
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urllib.parse.urlencode(params)}"
    logger.info(f"Initiated Google Drive OAuth redirect to: {redirect_uri}")
    return RedirectResponse(url=auth_url)


@router.get("/oauth2callback")
async def oauth2callback(request: Request, code: str = None, error: str = None, state: str = None):
    deps = request.app.state.deps
    settings = deps.settings

    if error:
        logger.warning(f"Google OAuth redirect error: {error}")
        return HTMLResponse(content=f"<h3>Authentication error from Google: {error}</h3>", status_code=400)
    if not code:
        logger.warning("Google OAuth redirect missing authorization code")
        return HTMLResponse(content="<h3>Missing authorization code from Google</h3>", status_code=400)

    client_id = settings.google_drive_client_id
    client_secret = settings.google_drive_client_secret
    if not client_id or not client_secret:
        return HTMLResponse(content="<h3>Google Drive credentials missing in .env</h3>", status_code=400)

    # Reconstruct redirect URI matching request hostname/port
    redirect_uri = str(request.base_url).rstrip("/") + "/oauth2callback"

    # Exchange authorization code for refresh token
    import urllib.parse
    import urllib.request
    import json

    body = urllib.parse.urlencode({
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }).encode()

    req = urllib.request.Request(
        "https://oauth2.googleapis.com/token",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err_body = e.read().decode(errors="ignore")
        logger.error(f"Error exchanging Google code: HTTP {e.code}\n{err_body}")
        return HTMLResponse(content=f"<h3>Error exchanging code: {err_body}</h3>", status_code=400)

    refresh_token = payload.get("refresh_token")
    if not refresh_token:
        logger.warning(f"No refresh_token returned in exchange payload: {payload}")
        return HTMLResponse(
            content="<h3>No refresh token returned. Revoke prior consent first at https://myaccount.google.com/permissions, then retry.</h3>",
            status_code=400
        )

    # Write the new refresh token to .env file in-place
    from pathlib import Path
    project_root = Path(__file__).resolve().parent.parent.parent
    env_path = project_root / ".env"

    try:
        # Backup .env first
        bak_path = env_path.with_suffix(".env.bak")
        bak_path.write_bytes(env_path.read_bytes())
    except Exception as e:
        logger.warning(f"Could not backup .env: {e}")

    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
        found = False
        out = []
        for ln in lines:
            if ln.startswith("GOOGLE_DRIVE_REFRESH_TOKEN="):
                out.append(f"GOOGLE_DRIVE_REFRESH_TOKEN={refresh_token}")
                found = True
            else:
                out.append(ln)
        if not found:
            out.append(f"GOOGLE_DRIVE_REFRESH_TOKEN={refresh_token}")

        env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
        logger.info(f"Google Drive refresh token successfully saved to {env_path}")
    except Exception as e:
        logger.exception("Failed to write new refresh token to .env")
        return HTMLResponse(content=f"<h3>Failed to update .env configuration file: {e}</h3>", status_code=500)

    # Hot-reload token in memory settings & reconnect the DriveLoader
    settings.google_drive_refresh_token = refresh_token
    try:
        deps.ingestion.loader.connect()
        logger.info("DriveLoader hot-reloaded and re-connected successfully")
        connection_status = "Google Drive connected successfully using new token!"
    except Exception as e:
        logger.exception("DriveLoader reconnection failed with the new token")
        connection_status = f"Warning: Token updated in .env, but Google Drive connection failed during hot-reload: {e}"

    success_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Google Drive Authorized</title>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            body {{
                font-family: 'Outfit', sans-serif;
                background-color: #0b0f19;
                color: #f3f4f6;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
                background-image: radial-gradient(circle at 50% 50%, rgba(99, 102, 241, 0.15) 0%, transparent 60%);
            }}
            .card {{
                background: rgba(22, 28, 45, 0.45);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 16px;
                padding: 2.5rem;
                max-width: 500px;
                width: 90%;
                text-align: center;
                backdrop-filter: blur(8px);
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
            }}
            h2 {{
                color: #10b981;
                font-size: 1.75rem;
                margin-top: 0;
            }}
            p {{
                color: #9ca3af;
                line-height: 1.5;
            }}
            .status {{
                margin: 1.5rem 0;
                padding: 1rem;
                background: rgba(16, 185, 129, 0.1);
                border: 1px solid rgba(16, 185, 129, 0.2);
                border-radius: 8px;
                font-size: 0.95rem;
                color: #34d399;
            }}
            .btn {{
                display: inline-block;
                background: #4f46e5;
                color: white;
                text-decoration: none;
                padding: 0.75rem 1.5rem;
                border-radius: 8px;
                font-weight: 600;
                margin-top: 1.5rem;
                transition: background-color 0.2s;
            }}
            .btn:hover {{
                background: #4338ca;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <h2>✔ Google Drive Authorized</h2>
            <p>The Google Drive OAuth refresh token has been successfully updated in your environment configuration.</p>
            <div class="status">{connection_status}</div>
            <a href="/admin" class="btn">Return to Admin Dashboard</a>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=success_html)
