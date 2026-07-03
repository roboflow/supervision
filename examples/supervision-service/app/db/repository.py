import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.db.database import _now_iso, get_connection


@dataclass
class UploadRecord:
    id: str
    original_filename: str
    stored_filename: str
    file_path: str
    file_size: int
    content_type: str | None
    created_at: str


@dataclass
class ProcessingJobRecord:
    id: str
    upload_id: str
    job_type: str
    status: str
    output_path: str | None
    parameters: dict[str, Any]
    error_message: str | None
    progress: int
    current_frame: int
    total_frames: int
    created_at: str
    completed_at: str | None
    original_filename: str | None = None


def create_upload(
    *,
    original_filename: str,
    stored_filename: str,
    file_path: Path,
    file_size: int,
    content_type: str | None,
) -> UploadRecord:
    """Insert an upload record and return it."""
    upload_id = uuid4().hex
    created_at = _now_iso()
    record = UploadRecord(
        id=upload_id,
        original_filename=original_filename,
        stored_filename=stored_filename,
        file_path=str(file_path),
        file_size=file_size,
        content_type=content_type,
        created_at=created_at,
    )
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO uploads (
                id, original_filename, stored_filename, file_path,
                file_size, content_type, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id,
                record.original_filename,
                record.stored_filename,
                record.file_path,
                record.file_size,
                record.content_type,
                record.created_at,
            ),
        )
        connection.commit()
    return record


def create_processing_job(
    *,
    upload_id: str,
    job_type: str,
    parameters: dict[str, Any],
) -> ProcessingJobRecord:
    """Create a pending processing job."""
    job_id = uuid4().hex
    created_at = _now_iso()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO processing_jobs (
                id, upload_id, job_type, status, output_path,
                parameters, error_message, progress, current_frame,
                total_frames, created_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                upload_id,
                job_type,
                "processing",
                None,
                json.dumps(parameters, ensure_ascii=False),
                None,
                0,
                0,
                0,
                created_at,
                None,
            ),
        )
        connection.commit()
    return ProcessingJobRecord(
        id=job_id,
        upload_id=upload_id,
        job_type=job_type,
        status="processing",
        output_path=None,
        parameters=parameters,
        error_message=None,
        progress=0,
        current_frame=0,
        total_frames=0,
        created_at=created_at,
        completed_at=None,
    )


def update_job_progress(
    job_id: str,
    *,
    progress: int,
    current_frame: int,
    total_frames: int,
) -> None:
    """Update frame progress for a running job."""
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE processing_jobs
            SET progress = ?, current_frame = ?, total_frames = ?
            WHERE id = ?
            """,
            (
                max(0, min(progress, 100)),
                current_frame,
                total_frames,
                job_id,
            ),
        )
        connection.commit()


def mark_job_completed(job_id: str, output_path: Path) -> None:
    """Mark a job as completed with its output file path."""
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE processing_jobs
            SET status = ?, output_path = ?, completed_at = ?,
                error_message = NULL, progress = 100
            WHERE id = ?
            """,
            ("completed", str(output_path), _now_iso(), job_id),
        )
        connection.commit()


def mark_job_failed(job_id: str, error_message: str) -> None:
    """Mark a job as failed with an error message."""
    with get_connection() as connection:
        connection.execute(
            """
            UPDATE processing_jobs
            SET status = ?, error_message = ?, completed_at = ?
            WHERE id = ?
            """,
            ("failed", error_message, _now_iso(), job_id),
        )
        connection.commit()


def list_uploads(limit: int = 50) -> list[UploadRecord]:
    """Return recent upload records."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, original_filename, stored_filename, file_path,
                   file_size, content_type, created_at
            FROM uploads
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [UploadRecord(**dict(row)) for row in rows]


def get_upload(upload_id: str) -> UploadRecord | None:
    """Return a single upload record by id."""
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT id, original_filename, stored_filename, file_path,
                   file_size, content_type, created_at
            FROM uploads
            WHERE id = ?
            """,
            (upload_id,),
        ).fetchone()
    if row is None:
        return None
    return UploadRecord(**dict(row))


def list_processing_jobs(limit: int = 50) -> list[ProcessingJobRecord]:
    """Return recent processing jobs with original filename."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                j.id, j.upload_id, j.job_type, j.status, j.output_path,
                j.parameters, j.error_message, j.progress, j.current_frame,
                j.total_frames, j.created_at, j.completed_at,
                u.original_filename
            FROM processing_jobs j
            LEFT JOIN uploads u ON u.id = j.upload_id
            ORDER BY j.created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [_row_to_job(row) for row in rows]


def get_processing_job(job_id: str) -> ProcessingJobRecord | None:
    """Return a single processing job by id."""
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                j.id, j.upload_id, j.job_type, j.status, j.output_path,
                j.parameters, j.error_message, j.progress, j.current_frame,
                j.total_frames, j.created_at, j.completed_at,
                u.original_filename
            FROM processing_jobs j
            LEFT JOIN uploads u ON u.id = j.upload_id
            WHERE j.id = ?
            """,
            (job_id,),
        ).fetchone()
    if row is None:
        return None
    return _row_to_job(row)


def _row_to_job(row: Any) -> ProcessingJobRecord:
    return ProcessingJobRecord(
        id=row["id"],
        upload_id=row["upload_id"],
        job_type=row["job_type"],
        status=row["status"],
        output_path=row["output_path"],
        parameters=json.loads(row["parameters"]),
        error_message=row["error_message"],
        progress=row["progress"],
        current_frame=row["current_frame"],
        total_frames=row["total_frames"],
        created_at=row["created_at"],
        completed_at=row["completed_at"],
        original_filename=row["original_filename"],
    )
