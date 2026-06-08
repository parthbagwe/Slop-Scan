# backend/app/models/feedback.py
#
# PURPOSE: Store user corrections so we can retrain the model.
# Every time a user says "this verdict was wrong", we store:
# - the original text
# - what our model predicted
# - what the user says it actually is
# - the feature scores at the time
#
# This becomes our "human-labeled" training data over time.

import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Float, Text, JSON, Boolean
from sqlalchemy.dialects.postgresql import UUID
from app.core.database import Base


class UserFeedback(Base):
    __tablename__ = "user_feedback"

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # The text that was analyzed
    content_text    = Column(Text, nullable=False)
    
    # What our model said
    predicted_label = Column(String, nullable=False)  # "AI" or "human"
    predicted_prob  = Column(Float, nullable=False)   # 0.0 to 1.0
    
    # What the user says it actually is
    true_label      = Column(String, nullable=False)  # "AI" or "human"
    
    # Was the model correct?
    is_correct      = Column(Boolean, nullable=False)
    
    # Feature scores at time of prediction (for analysis)
    feature_scores  = Column(JSON, nullable=True)
    
    # Used for retraining - has this been included in a training run?
    used_for_training = Column(Boolean, default=False)
    
    created_at      = Column(DateTime, default=datetime.utcnow)