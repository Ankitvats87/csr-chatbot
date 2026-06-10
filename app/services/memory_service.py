from typing import List

from app.models.message_model import Turn
from app.repositories.memory_repo import MemoryRepo
from app.utils.env import Settings


class MemoryService:
    def __init__(self, repo: MemoryRepo, settings: Settings):
        self.repo = repo
        self.settings = settings

    def recent(self, chat_id: int) -> List[Turn]:
        return self.repo.recent(chat_id, self.settings.memory_window)

    def record_user(self, chat_id: int, content: str) -> None:
        self.repo.append(chat_id, "user", content)

    def record_assistant(self, chat_id: int, content: str) -> None:
        self.repo.append(chat_id, "assistant", content)

    def clear(self, chat_id: int) -> None:
        self.repo.clear(chat_id)
