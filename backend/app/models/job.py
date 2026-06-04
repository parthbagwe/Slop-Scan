import uuid, enum
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Float, ForeignKey, Text, JSON, Enum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from app.core.database import Base

class ContentType(str, enum.Enum):
    TEXT = "text"
    # AUDIO = "audio"   ← add when building audio module
    # VIDEO = "video"   ← add when building video module

class JobStatus(str, enum.Enum):
    QUEUED     = "queued"
    PROCESSING = "processing"
    COMPLETED  = "completed"
    FAILED     = "failed"

class DetectionJob(Base):
    __tablename__ = "detection_jobs"
    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id      = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    content_type = Column(Enum(ContentType), nullable=False, default=ContentType.TEXT)
    content_text = Column(Text, nullable=True)
    status       = Column(Enum(JobStatus), nullable=False, default=JobStatus.QUEUED)
    error_message = Column(Text, nullable=True)
    created_at   = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    user   = relationship("User", back_populates="jobs")
    result = relationship("DetectionResult", back_populates="job", uselist=False)

class DetectionResult(Base):
    __tablename__ = "detection_results"
    id                  = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id              = Column(UUID(as_uuid=True), ForeignKey("detection_jobs.id"), nullable=False)
    ai_probability      = Column(Float, nullable=False)
    quality_score       = Column(Float, nullable=False)
    authenticity_score  = Column(Float, nullable=False)
    confidence          = Column(Float, nullable=False)
    explanation         = Column(JSON, nullable=True)
    feature_scores      = Column(JSON, nullable=True)
    model_version       = Column(String, nullable=False, default="1.0.0")
    created_at          = Column(DateTime, default=datetime.utcnow)
    job = relationship("DetectionJob", back_populates="result")