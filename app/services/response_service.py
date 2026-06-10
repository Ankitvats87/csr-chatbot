from dataclasses import dataclass
from typing import List, Optional, Tuple

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.utils.env import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str


class ResponseService:
    """Routes chat completions to OpenRouter first (default), falls back to OpenAI.
    Both providers use the OpenAI SDK (OpenRouter is OpenAI-API-compatible).
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._openrouter: Optional[OpenAI] = None
        self._openai: Optional[OpenAI] = None

        if settings.openrouter_api_key:
            self._openrouter = OpenAI(
                api_key=settings.openrouter_api_key,
                base_url=OPENROUTER_BASE_URL,
                default_headers={
                    "HTTP-Referer": settings.openrouter_referer,
                    "X-Title": settings.openrouter_app_title,
                },
            )
        if settings.openai_api_key:
            self._openai = OpenAI(api_key=settings.openai_api_key)

        if not self._openrouter and not self._openai:
            raise RuntimeError(
                "No chat provider configured. Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env."
            )

    def generate(self, messages: List[dict], model_override: Optional[str] = None) -> LLMResponse:
        """model_override replaces the configured model on the PRIMARY provider
        only (e.g. use gpt-4o for final answer synthesis while planner and
        extraction stay on the cheaper default)."""
        order = self._provider_order()
        if model_override and order:
            order = [(order[0][0], model_override)] + order[1:]
        last_err: Optional[Exception] = None
        for provider, model in order:
            try:
                text = self._call(provider, model, messages)
                return LLMResponse(text=text, provider=provider, model=model)
            except Exception as e:
                last_err = e
                logger.warning(
                    "llm provider failed, trying fallback",
                    extra={"provider": provider, "model": model, "err": str(e)},
                )
        raise RuntimeError(f"All LLM providers failed. Last error: {last_err}")

    def _provider_order(self) -> List[Tuple[str, str]]:
        order: List[Tuple[str, str]] = []
        primary = self.settings.llm_provider.lower()

        if primary == "openrouter":
            # Strict OpenRouter mode: never silently downgrade to OpenAI.
            if not self._openrouter:
                raise RuntimeError(
                    "LLM_PROVIDER=openrouter but OPENROUTER_API_KEY is not set. "
                    "Fill OPENROUTER_API_KEY in .env (and OPENROUTER_MODEL), then restart."
                )
            if not self.settings.openrouter_model:
                raise RuntimeError(
                    "OPENROUTER_MODEL is not set. Pick a model from https://openrouter.ai/models "
                    "and put it in .env (e.g. OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free)."
                )
            order.append(("openrouter", self.settings.openrouter_model))
            if self.settings.openrouter_fallback_to_openai and self._openai:
                order.append(("openai", self.settings.openai_chat_model))
        else:
            if self._openai:
                order.append(("openai", self.settings.openai_chat_model))
            if self._openrouter and self.settings.openrouter_model:
                order.append(("openrouter", self.settings.openrouter_model))
        return order

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4), reraise=True)
    def _call(self, provider: str, model: str, messages: List[dict]) -> str:
        client = self._openrouter if provider == "openrouter" else self._openai
        if client is None:
            raise RuntimeError(f"Provider {provider} not configured")
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=self.settings.llm_max_tokens,
        )
        return resp.choices[0].message.content or ""
