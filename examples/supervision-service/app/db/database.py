import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone

from app.config import DATABASE_PATH


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def _migrate(connection: sqlite3.Connection) -> None:
    columns = {
        row[1]
        for row in connection.execute("PRAGMA table_info(processing_jobs)").fetchall()
    }
    if "progress" not in columns:
        connection.execute(
            "ALTER TABLE processing_jobs ADD COLUMN progress INTEGER NOT NULL DEFAULT 0"
        )
    if "current_frame" not in columns:
        connection.execute(
            "ALTER TABLE processing_jobs ADD COLUMN current_frame INTEGER NOT NULL DEFAULT 0"
        )
    if "total_frames" not in columns:
        connection.execute(
            "ALTER TABLE processing_jobs ADD COLUMN total_frames INTEGER NOT NULL DEFAULT 0"
        )


def init_db() -> None:
    """Create database tables if they do not exist."""
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS uploads (
                id TEXT PRIMARY KEY,
                original_filename TEXT NOT NULL,
                stored_filename TEXT NOT NULL,
                file_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                content_type TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS processing_jobs (
                id TEXT PRIMARY KEY,
                upload_id TEXT NOT NULL,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL,
                output_path TEXT,
                parameters TEXT NOT NULL,
                error_message TEXT,
                progress INTEGER NOT NULL DEFAULT 0,
                current_frame INTEGER NOT NULL DEFAULT 0,
                total_frames INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                FOREIGN KEY (upload_id) REFERENCES uploads(id)
            );

            CREATE INDEX IF NOT EXISTS idx_uploads_created_at
                ON uploads(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_jobs_upload_id
                ON processing_jobs(upload_id);
            CREATE INDEX IF NOT EXISTS idx_jobs_created_at
                ON processing_jobs(created_at DESC);
            """
        )
        _migrate(connection)
        connection.commit()


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with row factory enabled."""
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
    finally:
        connection.close()
