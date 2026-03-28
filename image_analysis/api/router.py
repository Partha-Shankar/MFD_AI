import logging

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from image_analysis.schemas.response import AnalysisResponse

logger = logging.getLogger(__name__)

router = APIRouter()

_ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
_MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: Request, image: UploadFile) -> AnalysisResponse:
    """
    Accepts a multipart/form-data upload with field name "image".
    Validates content type and size, then runs the full forensics pipeline.
    """
    try:
        # Validate content type
        content_type = image.content_type or ""
        if content_type not in _ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=422,
                detail="Unsupported file type. Use JPEG, PNG, or WebP.",
            )

        # Read bytes
        image_bytes = await image.read()

        # Validate size
        if len(image_bytes) > _MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File too large. Max 20MB.",
            )

        pipeline = request.app.state.pipeline
        result = await pipeline.analyze(image_bytes)

        print(
            f"[analyze] verdict={result.verdict} "
            f"confidence={result.confidence:.2f}"
        )
        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[router] Unexpected error in /analyze: %s", exc)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(exc)}")
