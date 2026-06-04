from datetime import datetime
from celery import Celery
from app.core.config import settings
from app.core.database import SessionLocal
from app.models.job import DetectionJob, DetectionResult, JobStatus
import logging

logger = logging.getLogger(__name__)

celery_app = Celery("slop", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_serializer="json", accept_content=["json"],
    result_serializer="json", enable_utc=True,
    result_expires=86400, task_acks_late=True,
    worker_prefetch_multiplier=1,
)

@celery_app.task(name="tasks.process_text_detection", bind=True, max_retries=3, default_retry_delay=5)
def process_text_detection(self, job_id: str):
    db = SessionLocal()
    try:
        job = db.query(DetectionJob).filter(DetectionJob.id == job_id).first()
        if not job:
            return

        job.status = JobStatus.PROCESSING
        db.commit()

        from ml.text.detector import detect_ai_text
        data = detect_ai_text(job.content_text)

        result = DetectionResult(
            job_id=job.id,
            ai_probability=data["ai_probability"],
            quality_score=data["quality_score"],
            authenticity_score=data["authenticity_score"],
            confidence=data["confidence"],
            explanation=data["explanation"],
            feature_scores=data["feature_scores"],
            model_version=data["model_version"],
        )
        db.add(result)
        job.status       = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        db.commit()

        logger.info(f"Job {job_id} → {data['verdict']}")

    except Exception as exc:
        logger.error(f"Job {job_id} failed: {exc}")
        try:
            job = db.query(DetectionJob).filter(DetectionJob.id == job_id).first()
            if job:
                job.status        = JobStatus.FAILED
                job.error_message = str(exc)
                job.completed_at  = datetime.utcnow()
                db.commit()
        except Exception:
            pass
        raise self.retry(exc=exc)
    finally:
        db.close()