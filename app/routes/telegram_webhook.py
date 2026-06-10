import asyncio

from fastapi import APIRouter, BackgroundTasks, Header, HTTPException, Request

from app.utils.logger import get_logger
from app.utils.validators import constant_time_equals

logger = get_logger(__name__)

router = APIRouter()



@router.post("/webhook/telegram")
async def telegram_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
) -> dict:
    deps = request.app.state.deps
    settings = deps.settings

    # Hard-fail when the webhook secret is not configured. An empty secret
    # plus a missing header would otherwise compare equal and let anyone
    # forge updates.
    if not settings.telegram_webhook_secret:
        logger.error("rejected webhook: TELEGRAM_WEBHOOK_SECRET is not configured")
        raise HTTPException(status_code=503, detail="server misconfigured")

    if not constant_time_equals(
        x_telegram_bot_api_secret_token or "", settings.telegram_webhook_secret
    ):
        logger.warning("rejected webhook: bad secret header")
        raise HTTPException(status_code=403, detail="forbidden")

    try:
        update = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid json")

    # 1) Inline button presses (admin Approve / Deny on access requests).
    callback_query = update.get("callback_query")
    if callback_query:
        background_tasks.add_task(_handle_callback, deps, callback_query)
        return {"ok": True}

    # 2) Regular messages (or edits).
    message = update.get("message") or update.get("edited_message")
    if not message:
        return {"ok": True, "skipped": "non-message update"}

    chat = message.get("chat") or {}
    from_user = message.get("from") or {}
    chat_id = chat.get("id")
    username = from_user.get("username")
    text = (message.get("text") or "").strip()

    if not chat_id:
        return {"ok": True, "skipped": "no chat_id"}

    # /start handles the auth bootstrap; anyone can hit it.
    if text.lower().startswith("/start"):
        background_tasks.add_task(_handle_start, deps, chat_id, username)
        return {"ok": True}

    decision = deps.access.check(chat_id)
    if not decision.allowed:
        background_tasks.add_task(
            deps.telegram.send_message,
            chat_id,
            "You are not authorised to use this bot. Send /start to request access.",
        )
        logger.info("rejected unauthorised chat", extra={"chat_id": chat_id})
        return {"ok": True}

    # Non-text content from authorised users.
    if not text:
        background_tasks.add_task(
            deps.telegram.send_message,
            chat_id,
            "I can only handle text messages right now. Please type your question.",
        )
        return {"ok": True}

    # 3) Slash commands.
    if text.startswith("/"):
        cmd, _, rest = text.partition(" ")
        cmd = cmd.lower()
        if cmd == "/help":
            background_tasks.add_task(_send_help, deps, chat_id)
            return {"ok": True}
        if cmd == "/reset":
            deps.memory.clear(chat_id)
            background_tasks.add_task(deps.telegram.send_message, chat_id, "Memory cleared.")
            return {"ok": True}
        if cmd == "/memory":
            # Debug: show what the bot currently remembers for this chat.
            background_tasks.add_task(_send_memory_dump, deps, chat_id)
            return {"ok": True}

        # Admin-only commands
        if cmd in ("/grant", "/revoke", "/list", "/pending", "/ingest"):
            if not deps.access.is_admin(chat_id):
                background_tasks.add_task(
                    deps.telegram.send_message, chat_id, "Admin-only command."
                )
                return {"ok": True}
            background_tasks.add_task(_handle_admin_command, deps, chat_id, cmd, rest.strip())
            return {"ok": True}

    # 4) Default: route to RAG.
    background_tasks.add_task(_handle_query, deps, chat_id, text)
    return {"ok": True}


async def _send_help(deps, chat_id: int) -> None:
    text = (
        "I'm the Spark63 CSR assistant. Ask me anything about the knowledge base.\n\n"
        "User commands:\n"
        "/start, /help — this message\n"
        "/reset — clear my memory of our chat"
    )
    if deps.access.is_admin(chat_id):
        text += (
            "\n\nAdmin commands:\n"
            "/grant <chat_id|@username> — allow a user\n"
            "/revoke <chat_id|@username> — remove a user\n"
            "/list — show active users\n"
            "/pending — show pending requests\n"
            "/ingest — trigger an immediate Drive sync (best-effort)"
        )
    await deps.telegram.send_message(chat_id, text)


async def _handle_start(deps, chat_id: int, username: str | None) -> None:
    reply = await deps.access.handle_start(chat_id, username)
    await deps.telegram.send_message(chat_id, reply)


async def _send_memory_dump(deps, chat_id: int) -> None:
    turns = deps.memory.recent(chat_id)
    if not turns:
        await deps.telegram.send_message(chat_id, "Memory is empty for this chat.")
        return
    window = deps.settings.memory_window
    lines = [f"Memory (last {window} exchanges = up to {window * 2} messages):"]
    for i, t in enumerate(turns, 1):
        snippet = t.content if len(t.content) <= 200 else t.content[:200] + "…"
        lines.append(f"{i}. {t.role.upper()}: {snippet}")
    await deps.telegram.send_message(chat_id, "\n".join(lines))


async def _handle_admin_command(deps, chat_id: int, cmd: str, args: str) -> None:
    if cmd == "/grant":
        if not args:
            await deps.telegram.send_message(chat_id, "Usage: /grant <chat_id|@username>")
            return
        await deps.telegram.send_message(chat_id, deps.access.grant(target=args, granted_by=chat_id))
        return
    if cmd == "/revoke":
        if not args:
            await deps.telegram.send_message(chat_id, "Usage: /revoke <chat_id|@username>")
            return
        await deps.telegram.send_message(chat_id, deps.access.revoke(target=args))
        return
    if cmd == "/list":
        await deps.telegram.send_message(chat_id, deps.access.list_active())
        return
    if cmd == "/pending":
        await deps.telegram.send_message(chat_id, deps.access.list_pending())
        return
    if cmd == "/ingest":
        if not deps.settings.drive_configured():
            await deps.telegram.send_message(chat_id, "Google Drive is not configured in .env.")
            return
        await deps.telegram.send_message(chat_id, "Starting Google Drive sync...")
        try:
            counts = await asyncio.to_thread(deps.ingestion.sync_once)
            if "skipped" in counts:
                await deps.telegram.send_message(chat_id, "Drive sync skipped (not configured or running).")
                return
            msg = (
                f"Drive sync complete:\n"
                f"• Added: {counts.get('added', 0)}\n"
                f"• Updated: {counts.get('updated', 0)}\n"
                f"• Deleted: {counts.get('deleted', 0)}\n"
                f"• Failed: {counts.get('failed', 0)}"
            )
            await deps.telegram.send_message(chat_id, msg)
        except Exception as e:
            logger.exception("manual ingestion command failed", extra={"chat_id": chat_id, "err": str(e)})
            await deps.telegram.send_message(chat_id, f"Drive sync failed: {str(e)}")
        return


async def _handle_callback(deps, callback_query: dict) -> None:
    cb_id = callback_query.get("id", "")
    actor = (callback_query.get("from") or {}).get("id")
    data = callback_query.get("data", "")
    msg = callback_query.get("message") or {}
    if not actor:
        await deps.telegram.answer_callback_query(cb_id, "Unknown actor")
        return
    toast = await deps.access.handle_callback(actor_chat_id=actor, callback_data=data)
    await deps.telegram.answer_callback_query(cb_id, toast)
    # Edit the admin's original prompt so the buttons go away.
    chat_id = (msg.get("chat") or {}).get("id")
    message_id = msg.get("message_id")
    if chat_id and message_id:
        original = msg.get("text") or ""
        await deps.telegram.edit_message_text(chat_id, message_id, f"{original}\n\n→ {toast}")


async def _handle_query(deps, chat_id: int, text: str) -> None:
    await deps.telegram.send_typing(chat_id)
    try:
        result = await deps.rag.answer(chat_id, text)
        await deps.telegram.send_message(chat_id, result.answer)
    except Exception as e:
        logger.exception("query handler failed", extra={"chat_id": chat_id, "err": str(e)})
        await deps.telegram.send_message(
            chat_id,
            "Sorry — something went wrong while preparing your answer. Please try again in a moment.",
        )
