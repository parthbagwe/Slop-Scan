# backend/app/schemas/feedback.py

from pydantic import BaseModel
from uuid import UUID
from typing import Optional


class FeedbackRequest(BaseModel):
    # The text that was analyzed
    text: str
    
    # What the model predicted
    predicted_label: str   # "AI" or "human"
    predicted_prob:  float # 0.0 to 1.0
    
    # What the user says it actually is
    true_label: str        # "AI" or "human"
    
    # Feature scores from the original detection
    feature_scores: Optional[dict] = None


class FeedbackResponse(BaseModel):
    id:         UUID
    is_correct: bool
    message:    str