import logging

from fastapi import APIRouter, HTTPException

from image_analysis.feedback.store import save_feedback
from image_analysis.schemas.request import FeedbackRequest

logger = logging.getLogger(__name__)

feedback_router = APIRouter()


@feedback_router.post("/feedback")
async def feedback(payload: FeedbackRequest) -> dict:
    """
    Accept user feedback on a previous analysis verdict and persist it.
    """
    try:
        save_feedback(payload)
        return {
            "status": "saved",
            "message": "Thank you. Your feedback helps improve accuracy.",
        }
    except Exception as exc:
        logger.error("[feedback] Failed to save feedback: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(exc)}")
