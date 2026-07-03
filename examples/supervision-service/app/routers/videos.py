from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from app.config import DEFAULT_CONFIDENCE, DEFAULT_IOU
from app.db.repository import create_processing_job
from app.schemas.records import JobCreatedResponse
from app.services.job_runner import run_track_job
from app.services.upload_service import save_upload_file

router = APIRouter(prefix="/api/v1/videos", tags=["videos"])


@router.post("/track", response_model=JobCreatedResponse, status_code=202)
def track_uploaded_video(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="Input video file (mp4, mov, avi, etc.)")],
    confidence_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = DEFAULT_CONFIDENCE,
    iou_threshold: Annotated[float, Form(ge=0.0, le=1.0)] = DEFAULT_IOU,
) -> JobCreatedResponse:
    """Submit a tracking job and process it in the background."""
    try:
        upload_record = save_upload_file(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = create_processing_job(
        upload_id=upload_record.id,
        job_type="track",
        parameters={
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold,
        },
    )

    background_tasks.add_task(
        run_track_job,
        job.id,
        Path(upload_record.file_path),
        confidence_threshold,
        iou_threshold,
    )

    return JobCreatedResponse(job_id=job.id, upload_id=upload_record.id)
