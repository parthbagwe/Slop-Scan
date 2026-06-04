from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from uuid import UUID
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.job import DetectionJob, DetectionResult, JobStatus, ContentType
from app.schemas.detection import (
    TextDetectionRequest, JobCreatedResponse,
    DetectionResultResponse, DetectionHistoryResponse,
    FeatureScores, ExplanationItem
)

router = APIRouter(prefix="/detect", tags=["Detection"])

def _verdict_from_prob(prob: float) -> str:
    if prob >= 0.85: return "AI Generated"
    if prob >= 0.65: return "Likely AI"
    if prob >= 0.45: return "Uncertain"
    if prob >= 0.25: return "Likely Human"
    return "Human"

def _format(job: DetectionJob) -> DetectionResultResponse:
    r = job.result
    resp = DetectionResultResponse(
        job_id=job.id, status=job.status,
        content_type=job.content_type,
        completed_at=job.completed_at, error_message=job.error_message,
    )
    if r:
        resp.ai_probability     = r.ai_probability
        resp.quality_score      = r.quality_score
        resp.authenticity_score = r.authenticity_score
        resp.confidence         = r.confidence
        resp.verdict            = _verdict_from_prob(r.ai_probability)
        resp.model_version      = r.model_version
        resp.created_at         = r.created_at
        fs = r.feature_scores or {}
        resp.feature_scores = FeatureScores(**{k: fs.get(k) for k in FeatureScores.model_fields})
        if r.explanation:
            resp.explanation = [ExplanationItem(**e) for e in r.explanation]
    return resp

@router.post("/text", response_model=JobCreatedResponse, status_code=202)
async def submit_text(
    req: TextDetectionRequest,
    user: User = Depends(get_current_user),
    db:   Session = Depends(get_db),
):
    job = DetectionJob(user_id=user.id, content_type=ContentType.TEXT, content_text=req.text)
    db.add(job); db.commit(); db.refresh(job)
    from app.workers.tasks import process_text_detection
    process_text_detection.delay(str(job.id))
    return JobCreatedResponse(job_id=job.id, status=JobStatus.QUEUED)

@router.get("/jobs/{job_id}", response_model=DetectionResultResponse)
async def get_job(
    job_id: UUID,
    user:   User    = Depends(get_current_user),
    db:     Session = Depends(get_db),
):
    q = db.query(DetectionJob).filter(DetectionJob.id == job_id)
    if user.role != "admin":
        q = q.filter(DetectionJob.user_id == user.id)
    job = q.first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _format(job)

@router.get("/history", response_model=DetectionHistoryResponse)
async def history(
    page: int = 1, page_size: int = 20,
    user: User    = Depends(get_current_user),
    db:   Session = Depends(get_db),
):
    page_size = min(page_size, 100)
    q = db.query(DetectionJob).filter(
        DetectionJob.user_id == user.id,
        DetectionJob.status  == JobStatus.COMPLETED,
    ).order_by(DetectionJob.created_at.desc())
    total = q.count()
    jobs  = q.offset((page - 1) * page_size).limit(page_size).all()
    return DetectionHistoryResponse(jobs=[_format(j) for j in jobs], total=total, page=page, page_size=page_size)