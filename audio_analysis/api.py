"""
api.py — FastAPI REST server for the Deepfake Audio Detection System.

Endpoints:
    POST /analyze-audio   Upload an audio file and receive a forensic verdict.
    GET  /health          Service health check.
    GET  /docs            Auto-generated Swagger UI (FastAPI default).
"""

import io
import logging
import tempfile
import time
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from modules.metadata import MetadataAnalyzer
from modules.spectral import SpectralAnalyzer
from modules.temporal import TemporalAnalyzer
from modules.speaker import SpeakerConsistencyAnalyzer
from modules.fusion import VerdictEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

TARGET_SR = 16000
MAX_CONTENT_MB = 100
VERSION = "1.0.0"

app = FastAPI(
    title="Deepfake Audio Detection API",
    description=(
        "Signal-intelligence forensic pipeline for audio authenticity analysis. "
        "Detects AI-generated (TTS), voice-cloned, and spliced/edited audio without "
        "any ML training. Returns structured JSON with verdict, confidence, and anomalies."
    ),
    version=VERSION,
)

# CORS: allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module singletons (instantiated once at startup)
_meta_analyzer     = MetadataAnalyzer()
_spectral_analyzer = SpectralAnalyzer()
_temporal_analyzer = TemporalAnalyzer()
_speaker_analyzer  = SpeakerConsistencyAnalyzer()
_verdict_engine    = VerdictEngine()


# ── Startup warmup ────────────────────────────────────────────────────────────
@app.on_event("startup")
async def warmup() -> None:
    """Pre-warm librosa JIT compilation with a dummy signal."""
    logger.info("Warming up analysis pipeline...")
    try:
        sr = TARGET_SR
        dummy = np.random.randn(sr * 3).astype(np.float32) * 0.01
        _spectral_analyzer.analyze(dummy, sr)
        _temporal_analyzer.analyze(dummy, sr)
        logger.info("Pipeline warmup complete.")
    except Exception as e:
        logger.warning(f"Warmup failed (non-fatal): {e}")


# ── Request size limit middleware ─────────────────────────────────────────────
@app.middleware("http")
async def check_content_length(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_CONTENT_MB * 1024 * 1024:
        return JSONResponse(
            status_code=413,
            content={"error": f"File too large. Maximum is {MAX_CONTENT_MB} MB."},
        )
    return await call_next(request)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", summary="Service health check")
async def health():
    return {
        "status": "ok",
        "version": VERSION,
        "service": "deepfake-audio-detector",
    }


# ── Main analysis endpoint ────────────────────────────────────────────────────
@app.post(
    "/analyze-audio",
    summary="Analyze an audio file for deepfake indicators",
    response_description="Forensic verdict with scores, flags, and explanation",
)
async def analyze_audio(file: UploadFile = File(...)):
    """
    Upload an audio file (WAV, MP3, FLAC, OGG, M4A, etc.) and receive a
    comprehensive forensic verdict.

    Returns a JSON object with:
    - **verdict**: AUTHENTIC / AI GENERATED / VOICE CLONED / EDITED / AUTO TUNED
    - **confidence**: 0.0–1.0
    - **composite_score**: weighted suspicion score
    - **scores**: per-module scores
    - **anomalies**: human-readable anomaly descriptions
    - **flags**: machine-readable forensic flags
    - **explanation**: natural-language summary
    """
    start_time = time.time()

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    # Read uploaded bytes
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(contents) > MAX_CONTENT_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum is {MAX_CONTENT_MB} MB.",
        )

    # Write to a temp file (required for metadata analysis and librosa)
    suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # Load audio
        try:
            y_raw, sr_raw = librosa.load(tmp_path, sr=None, mono=True)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Cannot decode audio file: {e}")

        if len(y_raw) == 0:
            raise HTTPException(status_code=422, detail="Audio file contains no audio data.")

        # Resample to TARGET_SR
        if sr_raw != TARGET_SR:
            y = librosa.resample(y_raw, orig_sr=sr_raw, target_sr=TARGET_SR)
            sr = TARGET_SR
        else:
            y = y_raw
            sr = sr_raw

        duration = float(len(y) / sr)
        logger.info(
            f"Analyzing: {file.filename} | duration={duration:.1f}s | sr={sr} Hz"
        )

        # Run analysis modules
        try:
            metadata_result = _meta_analyzer.analyze(tmp_path)
            spectral_result = _spectral_analyzer.analyze(y, sr)
            temporal_result = _temporal_analyzer.analyze(y, sr)
            speaker_result  = _speaker_analyzer.analyze(y, sr)
        except Exception as e:
            logger.error(f"Analysis module error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Analysis error: {e}")

        file_info = {
            "path":             file.filename,
            "duration_seconds": round(duration, 2),
            "sample_rate":      sr_raw,
            "format":           suffix.lstrip(".").upper(),
            "file_size_bytes":  len(contents),
        }

        result = _verdict_engine.decide(
            metadata_result, spectral_result, temporal_result, speaker_result,
            file_info=file_info,
        )
        result["api_processing_time_seconds"] = round(time.time() - start_time, 3)

        return result

    finally:
        # Clean up temp file
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


# ── Run standalone ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
