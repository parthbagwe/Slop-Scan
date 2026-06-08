# ADD THIS to the bottom of tasks.py
# This is the Celery task that retrains the model when
# enough feedback has been collected.
from datetime import datetime
from celery import Celery
from app.core.config import settings
from app.core.database import SessionLocal
from app.models.job import DetectionJob, DetectionResult, JobStatus
from app.models.feedback import UserFeedback          # ← ADD THIS
import logging
logger = logging.getLogger(__name__)  
celery_app = Celery("slop", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    enable_utc=True,
    result_expires=86400,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


@celery_app.task(name="tasks.retrain_model", bind=True)
def retrain_model(self):
    """
    Retrain the RoBERTa model using accumulated user feedback.
    
    HOW IT WORKS:
    1. Load all unused feedback from PostgreSQL
    2. Combine with original training dataset
    3. Fine-tune the existing model further (not from scratch)
    4. Save new model version
    5. Mark feedback as used
    
    This is called "continual learning" or "online learning".
    The model gets smarter as more users correct it.
    """
    logger.info("Starting model retraining from user feedback...")
    db = SessionLocal()

    try:
        # Get all unused feedback
        feedbacks = db.query(UserFeedback).filter(
            UserFeedback.used_for_training == False
        ).all()

        if len(feedbacks) < 10:
            logger.info("Not enough feedback to retrain")
            return

        logger.info(f"Retraining with {len(feedbacks)} feedback samples")

        # Prepare training data from feedback
        texts  = [f.content_text for f in feedbacks]
        labels = [1 if f.true_label == "AI" else 0 for f in feedbacks]

        # Run the retraining script
        from ml.text.retrain import retrain_with_feedback
        success = retrain_with_feedback(texts, labels)

        if success:
            # Mark all feedback as used
            for f in feedbacks:
                f.used_for_training = True
            db.commit()
            logger.info("Retraining complete — model updated")
        else:
            logger.error("Retraining failed")

    except Exception as e:
        logger.error(f"Retrain task failed: {e}")
        raise self.retry(exc=e)
    finally:
        db.close()