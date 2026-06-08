# backend/app/api/feedback.py
#
# PURPOSE: Receive user feedback and store it.
# Also checks if we have enough feedback to trigger retraining.
#
# RETRAINING TRIGGER:
# When we get 50 new feedback entries that haven't been used
# for training yet, we automatically kick off a retraining job.
# 50 is a reasonable threshold — enough signal, not too frequent.

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.database import get_db
from app.core.security import get_current_user
from app.models.user import User
from app.models.feedback import UserFeedback
from app.schemas.feedback import FeedbackRequest, FeedbackResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback", tags=["Feedback"])

# How many feedback entries before we retrain
RETRAIN_THRESHOLD = 50


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(
    req: FeedbackRequest,
    db:  Session = Depends(get_db),
):
    """
    Store user feedback about a detection result.
    No auth required — we want as much feedback as possible.
    
    After storing, check if we have enough to retrain.
    """
    is_correct = (req.predicted_label == req.true_label)

    feedback = UserFeedback(
        content_text    = req.text,
        predicted_label = req.predicted_label,
        predicted_prob  = req.predicted_prob,
        true_label      = req.true_label,
        is_correct      = is_correct,
        feature_scores  = req.feature_scores,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)

    logger.info(
        f"Feedback stored: predicted={req.predicted_label}, "
        f"actual={req.true_label}, correct={is_correct}"
    )

    # Check if we should trigger retraining
    unused_count = db.query(UserFeedback).filter(
        UserFeedback.used_for_training == False
    ).count()

    if unused_count >= RETRAIN_THRESHOLD:
        logger.info(f"{unused_count} feedbacks ready — triggering retrain")
        from app.workers.tasks import retrain_model
        retrain_model.delay()

    return FeedbackResponse(
        id=feedback.id,
        is_correct=is_correct,
        message=(
            "Thanks! Your feedback helps improve the model." 
            if not is_correct 
            else "Thanks for confirming!"
        )
    )


@router.get("/stats")
async def feedback_stats(db: Session = Depends(get_db)):
    """
    Public endpoint showing model accuracy based on user feedback.
    Great for showing on the website — builds trust.
    """
    total   = db.query(UserFeedback).count()
    correct = db.query(UserFeedback).filter(
        UserFeedback.is_correct == True
    ).count()

    return {
        "total_feedback":   total,
        "correct":          correct,
        "incorrect":        total - correct,
        "accuracy":         round(correct / total, 3) if total > 0 else None,
        "untrained_samples": db.query(UserFeedback).filter(
            UserFeedback.used_for_training == False
        ).count()
    }