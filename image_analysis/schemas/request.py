from fastapi import UploadFile
from pydantic import BaseModel


class AnalysisRequest:
    image: UploadFile


class FeedbackRequest(BaseModel):
    image_hash: str
    user_verdict: str
    original_verdict: str
    comment: str | None = None
