"""
video_analysis/router.py
========================
FastAPI router for the ``/video/analyze`` endpoint.

Responsibility
--------------
*   Accept a multipart video upload.
*   Save it to a OS-managed temporary file.
*   Call :func:`video_analysis.pipeline.run_pipeline` to obtain the verdict.
*   Return a structured JSON response.
*   Clean up the temporary file regardless of outcome.


"""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Optional

from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from video_analysis.pipeline import run_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/video", tags=["Video Analysis"])


@router.post("/analyze", summary="Deepfake video analysis endpoint")
async def analyze_video(
    file: UploadFile = File(..., description="Video file to analyse (MP4 / AVI / MOV)"),
    x_bypass_code: Optional[str] = Header(
        None,
        description="Internal testing bypass: 'real' or 'ai'.  Do not expose publicly.",
    ),
) -> JSONResponse:
    """Accepts a video upload, runs the forensic pipeline, and returns a verdict.

    Request
    -------
    ``Content-Type: multipart/form-data``

    Body fields:

    +-----------+----------+-----------------------------------------------+
    | Field     | Type     | Description                                   |
    +===========+==========+===============================================+
    | file      | binary   | Video file (MP4, AVI, MOV, …)                 |
    +-----------+----------+-----------------------------------------------+

    Response (200 OK)
    -----------------
    .. code-block:: json

        {
            "verdict": "Video Likely Real (20% fake probability)",
            "filename": "clip.mp4"
        }

    Raises
    ------
    HTTP 422
        If no file is provided.
    HTTP 500
        If the analysis pipeline encounters an unrecoverable error.
    """
    if not file or not file.filename:
        raise HTTPException(status_code=422, detail="No video file provided.")

    # ── Persist upload to a temporary file ──────────────────────────────────
    suffix = os.path.splitext(file.filename)[-1] or ".mp4"
    tmp_path: str = ""

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            chunk_size = 1024 * 256  # 256 KB chunks
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                tmp.write(chunk)

        logger.info("[router] Saved upload to: %s", tmp_path)

        # ── Run the detection pipeline ───────────────────────────────────────
        verdict = run_pipeline(tmp_path, bypass_code=x_bypass_code)

        logger.info("[router] Pipeline returned verdict: %s", verdict)

        return JSONResponse(
            content={
                "verdict": verdict,
                "filename": file.filename,
            }
        )

    except FileNotFoundError as exc:
        logger.error("[router] File not found error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    except Exception as exc:
        logger.exception("[router] Unhandled pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail="Video analysis failed.") from exc

    finally:
        # ── Always clean up the temp file ───────────────────────────────────
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logger.info("[router] Cleaned up temp file: %s", tmp_path)
            except OSError as exc:
                logger.warning("[router] Could not remove temp file %s: %s", tmp_path, exc)
