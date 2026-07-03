from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.db.repository import (
    get_processing_job,
    get_upload,
    list_processing_jobs,
    list_uploads,
)
from app.schemas.records import (
    ProcessingJobListResponse,
    ProcessingJobResponse,
    UploadListResponse,
    UploadResponse,
)

router = APIRouter(prefix="/api/v1/records", tags=["records"])


def _to_upload_response(record: object) -> UploadResponse:
    return UploadResponse.model_validate(record.__dict__)


def _to_job_response(record: object) -> ProcessingJobResponse:
    return ProcessingJobResponse.model_validate(record.__dict__)


@router.get("/uploads", response_model=UploadListResponse)
def get_upload_records(
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> UploadListResponse:
    """List stored upload records."""
    items = [_to_upload_response(record) for record in list_uploads(limit=limit)]
    return UploadListResponse(items=items, total=len(items))


@router.get("/uploads/{upload_id}", response_model=UploadResponse)
def get_upload_record(upload_id: str) -> UploadResponse:
    """Return a single upload record."""
    record = get_upload(upload_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Upload record not found.")
    return _to_upload_response(record)


@router.get("/uploads/{upload_id}/file")
def download_upload_file(upload_id: str) -> FileResponse:
    """Download the original uploaded video file."""
    record = get_upload(upload_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Upload record not found.")

    file_path = Path(record.file_path)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Upload file not found on disk.")

    return FileResponse(
        path=file_path,
        media_type=record.content_type or "video/mp4",
        filename=record.original_filename,
    )


@router.get("/jobs", response_model=ProcessingJobListResponse)
def get_processing_jobs(
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> ProcessingJobListResponse:
    """List video processing job records."""
    items = [_to_job_response(record) for record in list_processing_jobs(limit=limit)]
    return ProcessingJobListResponse(items=items, total=len(items))


@router.get("/jobs/{job_id}", response_model=ProcessingJobResponse)
def get_processing_job_record(job_id: str) -> ProcessingJobResponse:
    """Return a single processing job record."""
    record = get_processing_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Processing job not found.")
    return _to_job_response(record)


@router.get("/jobs/{job_id}/file")
def download_job_result(job_id: str) -> FileResponse:
    """Download the processed output video for a job."""
    record = get_processing_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Processing job not found.")
    if record.status != "completed" or not record.output_path:
        raise HTTPException(status_code=404, detail="Processed file is not available.")

    file_path = Path(record.output_path)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="Processed file not found on disk.")

    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=file_path.name,
    )
