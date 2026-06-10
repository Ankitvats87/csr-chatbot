import io
import os
from dataclasses import dataclass
from typing import List, Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from app.utils.env import Settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Native Google Doc mime types we will export to text.
GOOGLE_EXPORT_MAP = {
    "application/vnd.google-apps.document": ("text/plain", "txt"),
    "application/vnd.google-apps.spreadsheet": ("text/csv", "csv"),
}

# Supported third-party mime types.
SUPPORTED_MIMES = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/plain": "txt",
    "text/csv": "csv",
}


@dataclass
class DriveFile:
    id: str
    name: str
    mime_type: str
    modified_time: str
    size_bytes: int

    @property
    def is_google_native(self) -> bool:
        return self.mime_type in GOOGLE_EXPORT_MAP

    def effective_ext(self) -> Optional[str]:
        if self.is_google_native:
            return GOOGLE_EXPORT_MAP[self.mime_type][1]
        return SUPPORTED_MIMES.get(self.mime_type)


class DriveLoader:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._service = None
        self._folder_id = settings.google_drive_folder_id
        self._max_bytes = settings.ingestion_max_file_mb * 1024 * 1024

    def connect(self) -> None:
        creds = Credentials(
            token=None,
            refresh_token=self.settings.google_drive_refresh_token,
            client_id=self.settings.google_drive_client_id,
            client_secret=self.settings.google_drive_client_secret,
            token_uri="https://oauth2.googleapis.com/token",
            scopes=SCOPES,
        )
        creds.refresh(Request())
        self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
        logger.info("google drive connected", extra={"folder_id": self._folder_id})

    def list_files(self) -> List[DriveFile]:
        if self._service is None:
            self.connect()
        files: List[DriveFile] = []
        page_token: Optional[str] = None
        query = (
            f"'{self._folder_id}' in parents and trashed = false and ("
            + " or ".join(
                f"mimeType = '{m}'" for m in list(SUPPORTED_MIMES.keys()) + list(GOOGLE_EXPORT_MAP.keys())
            )
            + ")"
        )
        while True:
            resp = (
                self._service.files()
                .list(
                    q=query,
                    fields="nextPageToken, files(id, name, mimeType, modifiedTime, size)",
                    pageSize=200,
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            for f in resp.get("files", []):
                files.append(
                    DriveFile(
                        id=f["id"],
                        name=f["name"],
                        mime_type=f["mimeType"],
                        modified_time=f["modifiedTime"],
                        size_bytes=int(f.get("size") or 0),
                    )
                )
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
        return files

    def download(self, file: DriveFile, dest_dir: str) -> Optional[str]:
        if file.size_bytes and file.size_bytes > self._max_bytes:
            logger.warning(
                "skipping oversized file",
                extra={"file_id": file.id, "file_name": file.name, "size_bytes": file.size_bytes},
            )
            return None
        if self._service is None:
            self.connect()
        os.makedirs(dest_dir, exist_ok=True)
        ext = file.effective_ext()
        if not ext:
            logger.warning("unsupported mime", extra={"mime": file.mime_type, "file_name": file.name})
            return None
        local_path = os.path.join(dest_dir, f"{file.id}.{ext}")

        if file.is_google_native:
            export_mime = GOOGLE_EXPORT_MAP[file.mime_type][0]
            request = self._service.files().export_media(fileId=file.id, mimeType=export_mime)
        else:
            request = self._service.files().get_media(fileId=file.id)

        buf = io.FileIO(local_path, "wb")
        try:
            downloader = MediaIoBaseDownload(buf, request, chunksize=1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        finally:
            buf.close()
        return local_path
