from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.config import UPLOAD_DIR
from app.db.repository import UploadRecord, create_upload


def save_upload_file(file: UploadFile) -> UploadRecord:
    """Persist an uploaded video under ``uploads/`` and store metadata.

    Args:
        file: Incoming multipart upload.

    Returns:
        Created upload database record.

    Raises:
        ValueError: If filename is missing or file is empty.
    """
    if not file.filename:
        raise ValueError("Filename is required.")

    content = file.file.read()
    if not content:
        raise ValueError("Uploaded file is empty.")

    suffix = Path(file.filename).suffix or ".mp4"
    stored_filename = f"{uuid4().hex}{suffix}"
    upload_path = UPLOAD_DIR / stored_filename
    upload_path.write_bytes(content)

    return create_upload(
        original_filename=file.filename,
        stored_filename=stored_filename,
        file_path=upload_path,
        file_size=len(content),
        content_type=file.content_type,
    )
