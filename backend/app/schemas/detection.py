from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime
from uuid import UUID
from app.models.job import ContentType, JobStatus

class TextDetectionRequest(BaseModel):
    text: str = Field(..., min_length=50, max_length=50_000)

    @validator("text")
    def not_blank(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be blank")
        return v.strip()

class JobCreatedResponse(BaseModel):
    job_id:  UUID
    status:  JobStatus
    message: str = "Queued"

class FeatureScores(BaseModel):
    perplexity:           Optional[float] = None
    burstiness:           Optional[float] = None
    vocabulary_diversity: Optional[float] = None
    information_density:  Optional[float] = None
    roberta_score:        Optional[float] = None
    avg_sentence_length:  Optional[float] = None
    type_token_ratio:     Optional[float] = None

class ExplanationItem(BaseModel):
    feature:     str
    impact:      float
    description: str

class DetectionResultResponse(BaseModel):
    job_id:              UUID
    status:              JobStatus
    content_type:        ContentType
    ai_probability:      Optional[float] = None
    quality_score:       Optional[float] = None
    authenticity_score:  Optional[float] = None
    confidence:          Optional[float] = None
    verdict:             Optional[str]   = None
    feature_scores:      Optional[FeatureScores]      = None
    explanation:         Optional[list[ExplanationItem]] = None
    model_version:       Optional[str]      = None
    created_at:          Optional[datetime] = None
    completed_at:        Optional[datetime] = None
    error_message:       Optional[str]      = None
    class Config: from_attributes = True

class DetectionHistoryResponse(BaseModel):
    jobs:      list[DetectionResultResponse]
    total:     int
    page:      int
    page_size: int