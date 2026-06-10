"""Reset the web-admin dashboard credentials by clearing them from SQLite.

After this runs, the dashboard falls back to ADMIN_USERNAME / ADMIN_PASSWORD
from .env. Set those to non-default values before running, otherwise the
dashboard will refuse to authenticate (by design).

Usage on the VPS (inside the app container):
    docker compose exec app python scripts/reset_admin_password.py

Or locally:
    python scripts/reset_admin_password.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.sqlite_client import SQLiteClient
from app.utils.env import get_settings


def main() -> int:
    settings = get_settings()
    db = SQLiteClient(settings.sqlite_path)
    db.connect()
    db.execute("DELETE FROM web_admin_config WHERE key IN ('username', 'password_hash')")
    db.close()
    print("Cleared web_admin_config. Dashboard now uses ADMIN_USERNAME / ADMIN_PASSWORD from .env.")
    print("Make sure those are NOT 'admin' / 'admin' or the dashboard will stay disabled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
