from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.core.config import settings
from app.core.database import engine, Base
from app.api import auth, detection
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield

app = FastAPI(title="AI Slop Detector API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

app.include_router(auth.router,      prefix="/api")
app.include_router(detection.router, prefix="/api")

@app.get("/health")
def health():
    return {"status": "healthy", "version": "1.0.0"}

# Add this import at the top with the others
from app.api import auth, detection, feedback

# Add this line after the existing include_router lines
app.include_router(feedback.router, prefix="/api")