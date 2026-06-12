from typing import List

import httpx

from app.utils.env import Settings
from app.utils.logger import get_logger
from app.utils.telegram_format import format_for_telegram_html, html_to_plain
from app.utils.validators import split_telegram_message

logger = get_logger(__name__)

TELEGRAM_API = "https://api.telegram.org"


class TelegramService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = httpx.AsyncClient(
            base_url=f"{TELEGRAM_API}/bot{settings.telegram_bot_token}",
            timeout=httpx.Timeout(15.0, connect=5.0),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, method: str, payload: dict) -> dict | None:
        try:
            resp = await self._client.post(f"/{method}", json=payload)
            resp.raise_for_status()
            data = resp.json()

            if not data.get("ok", False):
                logger.error(
                    "telegram api returned ok=false",
                    extra={
                        "method": method,
                        "payload": payload,
                        "response": data,
                    },
                )
                return data

            return data

        except httpx.HTTPStatusError as e:
            logger.error(
                "telegram api http error",
                extra={
                    "method": method,
                    "payload": payload,
                    "status": e.response.status_code,
                    "body": e.response.text,
                },
            )
        except Exception as e:
            logger.exception(
                "telegram api exception",
                extra={"method": method, "payload": payload, "err": str(e)},
            )
        return None

    async def send_message(self, chat_id: int, text: str) -> None:
        """Send a text message. Long messages are split on paragraph boundaries
        to stay under Telegram's 4096-char limit. Sent as HTML (real bold) with
        a plain-text retry if Telegram rejects the markup. All errors are caught
        and logged so a failed send never crashes the caller (we run inside
        BackgroundTasks).
        """
        if not text:
            return
        formatted = format_for_telegram_html(text)
        for part in split_telegram_message(formatted):
            resp = await self._post(
                "sendMessage",
                {
                    "chat_id": chat_id,
                    "text": part,
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
            )
            if resp is None or not resp.get("ok", False):
                await self._post(
                    "sendMessage",
                    {
                        "chat_id": chat_id,
                        "text": html_to_plain(part),
                        "disable_web_page_preview": True,
                    },
                )


    async def send_typing(self, chat_id: int) -> None:
        await self._post(
            "sendChatAction",
            {"chat_id": chat_id, "action": "typing"},
        )

    async def send_message_with_markup(self, chat_id: int, text: str, reply_markup: dict) -> None:
        await self._post(
            "sendMessage",
            {
                "chat_id": chat_id,
                "text": text,
                "reply_markup": reply_markup,
                "disable_web_page_preview": True,
            },
        )

    async def answer_callback_query(self, callback_query_id: str, text: str = "") -> None:
        await self._post(
            "answerCallbackQuery",
            {"callback_query_id": callback_query_id, "text": text[:200]},
        )

    async def edit_message_text(self, chat_id: int, message_id: int, text: str) -> None:
        await self._post(
            "editMessageText",
            {"chat_id": chat_id, "message_id": message_id, "text": text},
        )