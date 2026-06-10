from dataclasses import dataclass
from typing import List, Optional

from app.repositories.access_repo import AccessRepo
from app.services.telegram_service import TelegramService
from app.utils.env import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AccessDecision:
    allowed: bool
    reason: str  # "admin" | "static_allow" | "dynamic_allow" | "bootstrap" | "denied"


class AccessService:
    """Admin-managed allowlist with inline approve/deny buttons.

    Bootstrap rule: if no admins are configured in .env AND no dynamic admin
    exists yet, the very first user who sends /start is auto-claimed as the
    first admin. This lets you ship the bot without baking your chat_id into
    .env first.
    """

    def __init__(self, repo: AccessRepo, telegram: TelegramService, settings: Settings):
        self.repo = repo
        self.telegram = telegram
        self.settings = settings
        # In-memory cache of bootstrap admins promoted at runtime (so we don't
        # need to round-trip the DB on every message). Persisted alongside
        # static admins for the duration of the process.
        self._runtime_admins: set[int] = set()

    # ---------- gating ----------
    def is_admin(self, chat_id: int) -> bool:
        return (
            chat_id in self.settings.admin_chat_ids
            or chat_id in self._runtime_admins
        )

    def check(self, chat_id: int) -> AccessDecision:
        if self.is_admin(chat_id):
            return AccessDecision(True, "admin")
        if chat_id in self.settings.allowed_chat_ids:
            return AccessDecision(True, "static_allow")
        if self.repo.is_active(chat_id):
            return AccessDecision(True, "dynamic_allow")
        return AccessDecision(False, "denied")

    def _has_any_admin(self) -> bool:
        return bool(self.settings.admin_chat_ids) or bool(self._runtime_admins)

    # ---------- onboarding ----------
    async def handle_start(self, chat_id: int, username: Optional[str]) -> str:
        """Returns the text the bot should reply with to the user."""
        # Bootstrap: first /start with no configured admin auto-claims admin.
        if not self._has_any_admin():
            self._runtime_admins.add(chat_id)
            self.repo.grant(chat_id, username, granted_by=chat_id)
            logger.info("bootstrap admin claimed", extra={"chat_id": chat_id, "username": username})
            return (
                "You are the first user and have been registered as the bot administrator.\n\n"
                "Add your chat_id to ADMIN_CHAT_IDS in .env to make this permanent across restarts.\n\n"
                "Commands you can use:\n"
                "/grant <chat_id> — allow a user\n"
                "/revoke <chat_id> — remove a user\n"
                "/list — show active users\n"
                "/pending — show users awaiting approval"
            )

        decision = self.check(chat_id)
        if decision.allowed:
            return (
                "Hi! I'm the Spark63 CSR assistant. Ask me anything about the knowledge base.\n\n"
                "Commands:\n/start, /help — this message\n/reset — clear my memory of our chat"
            )

        # New / pending user: record + notify admins
        is_new = self.repo.record_request(chat_id, username)
        if is_new:
            await self._notify_admins_of_request(chat_id, username)
        return "Access pending. The administrator has been notified and will review your request shortly."

    async def _notify_admins_of_request(self, chat_id: int, username: Optional[str]) -> None:
        admin_ids = list(self.settings.admin_chat_ids) + list(self._runtime_admins)
        if not admin_ids:
            return
        label = f"@{username}" if username else f"chat_id {chat_id}"
        text = (
            f"Access request from {label}\n"
            f"chat_id: {chat_id}\n\n"
            f"Approve or deny:"
        )
        reply_markup = {
            "inline_keyboard": [
                [
                    {"text": "Approve", "callback_data": f"access:approve:{chat_id}"},
                    {"text": "Deny", "callback_data": f"access:deny:{chat_id}"},
                ]
            ]
        }
        for admin_id in set(admin_ids):
            try:
                await self.telegram.send_message_with_markup(admin_id, text, reply_markup)
            except Exception as e:
                logger.warning("failed to notify admin", extra={"admin_id": admin_id, "err": str(e)})

    # ---------- callback (inline buttons) ----------
    async def handle_callback(
        self, *, actor_chat_id: int, callback_data: str
    ) -> str:
        """Returns a short toast string for answerCallbackQuery."""
        if not self.is_admin(actor_chat_id):
            return "You are not authorised."
        try:
            _, action, target_str = callback_data.split(":")
            target = int(target_str)
        except Exception:
            return "Bad callback."
        if action == "approve":
            self.repo.grant(target, username=None, granted_by=actor_chat_id)
            self.repo.set_request_decision(target, "approved")
            try:
                await self.telegram.send_message(
                    target,
                    "Your access has been approved. Send me a question to get started.",
                )
            except Exception:
                pass
            return f"Approved {target}"
        if action == "deny":
            self.repo.set_request_decision(target, "denied")
            return f"Denied {target}"
        return "Unknown action."

    # ---------- admin commands ----------
    def grant(self, *, target: str, granted_by: int) -> str:
        chat_id = self._resolve_target(target)
        if chat_id is None:
            return f"Could not resolve {target}. Use a numeric chat_id."
        self.repo.grant(chat_id, username=target if target.startswith("@") else None, granted_by=granted_by)
        self.repo.set_request_decision(chat_id, "approved")
        return f"Granted access to {chat_id}."

    def revoke(self, *, target: str) -> str:
        chat_id = self._resolve_target(target)
        if chat_id is None:
            return f"Could not resolve {target}. Use a numeric chat_id."
        ok = self.repo.revoke(chat_id)
        return f"Revoked {chat_id}." if ok else f"{chat_id} was not currently active."

    def list_active(self) -> str:
        entries = self.repo.list_active()
        if not entries:
            return "No active users."
        lines = ["Active users:"]
        for e in entries:
            who = f"@{e.username}" if e.username else "(no username)"
            lines.append(f"• {e.chat_id}  {who}  granted {e.granted_at}")
        return "\n".join(lines)

    def list_pending(self) -> str:
        reqs = self.repo.pending_requests()
        if not reqs:
            return "No pending requests."
        lines = ["Pending requests:"]
        for r in reqs:
            who = f"@{r.username}" if r.username else "(no username)"
            lines.append(f"• {r.chat_id}  {who}  first seen {r.first_seen}")
        lines.append("\nApprove with: /grant <chat_id>")
        return "\n".join(lines)

    def _resolve_target(self, target: str) -> Optional[int]:
        target = target.strip()
        if not target:
            return None
        if target.startswith("@"):
            return self.repo.resolve_username(target)
        try:
            return int(target)
        except ValueError:
            return None
