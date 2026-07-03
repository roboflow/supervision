from typing import Any

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Stored upload record."""

    id: str
    original_filename: str
    stored_filename: str
    file_path: str
    file_size: int
    content_type: str | None
    created_at: str


class ProcessingJobResponse(BaseModel):
    """Processing job record."""

    id: str
    upload_id: str
    job_type: str
    status: str
    output_path: str | None
    parameters: dict[str, Any]
    error_message: str | None
    progress: int = Field(ge=0, le=100)
    current_frame: int = Field(ge=0)
    total_frames: int = Field(ge=0)
    created_at: str
    completed_at: str | None
    original_filename: str | None = None


class JobCreatedResponse(BaseModel):
    """Response after submitting an async processing job."""

    job_id: str
    upload_id: str
    status: str = "processing"


class UploadListResponse(BaseModel):
    """Paginated upload list."""

    items: list[UploadResponse]
    total: int = Field(description="Number of items returned")


class ProcessingJobListResponse(BaseModel):
    """Paginated processing job list."""

    items: list[ProcessingJobResponse]
    total: int = Field(description="Number of items returned")
