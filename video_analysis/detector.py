"""
video_analysis/detector.py
==========================
Core deepfake video detection engine.

Loads two HuggingFace frame-classification models and scores each sampled
frame for authenticity.  The final verdict is built from three independent
signal axes:

    1. Neural ensemble vote   (frame-level deepfake classification)
    2. Frequency artifacts    (FFT magnitude anomalies)
    3. Temporal coherence     (inter-frame motion delta)

Only ``detect_fake_video`` is the public surface of this module.
Everything else is implementation detail.
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model registry  (lazy, module-level singletons)
# ---------------------------------------------------------------------------

_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
)

_vid_stuff1: Optional[Tuple] = None   # (processor, model) for M1
_vid_stuff2: Optional[Tuple] = None   # (processor, model) for M2


def get_video_model1() -> Tuple:
    """Return the (processor, model) pair for the primary deepfake detector.

    Model is loaded lazily on first call and cached for the lifetime of the
    process.  Weights are pulled from the HuggingFace Hub and cached locally
    under ``~/.cache/huggingface``.

    Returns
    -------
    tuple
        ``(AutoImageProcessor, AutoModelForImageClassification)``
    """
    global _vid_stuff1
    if _vid_stuff1 is None:
        logger.info("[detector] Loading primary video classification model …")
        processor = AutoImageProcessor.from_pretrained(
            "prithivMLmods/deepfake-detector-model-v1"
        )
        model = AutoModelForImageClassification.from_pretrained(
            "prithivMLmods/deepfake-detector-model-v1"
        )
        model.eval()
        _vid_stuff1 = (processor, model)
        logger.info("[detector] Primary model loaded ✓")
    return _vid_stuff1


def get_video_model2() -> Tuple:
    """Return the (processor, model) pair for the secondary deepfake detector.

    Returns
    -------
    tuple
        ``(AutoImageProcessor, AutoModelForImageClassification)``
    """
    global _vid_stuff2
    if _vid_stuff2 is None:
        logger.info("[detector] Loading secondary video classification model …")
        processor = AutoImageProcessor.from_pretrained(
            "dima806/deepfake_vs_real_image_detection"
        )
        model = AutoModelForImageClassification.from_pretrained(
            "dima806/deepfake_vs_real_image_detection"
        )
        model.eval()
        _vid_stuff2 = (processor, model)
        logger.info("[detector] Secondary model loaded ✓")
    return _vid_stuff2


def clear_video_models() -> None:
    """Release all cached model weights from memory.

    Should be called after inference is complete to free VRAM / RAM before
    the next request is serviced.
    """
    global _vid_stuff1, _vid_stuff2
    _vid_stuff1 = _vid_stuff2 = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("[detector] Model weights released from memory.")


# ---------------------------------------------------------------------------
# Signal extractors
# ---------------------------------------------------------------------------

def frequency_artifacts(frame: np.ndarray) -> float:
    """Compute the mean log-magnitude of the 2-D FFT of a BGR frame.

    High values indicate unusual high-frequency energy patterns commonly
    produced by GAN upsamplers and frame-interpolation networks.

    Parameters
    ----------
    frame:
        A (H, W, 3) uint8 BGR frame as returned by ``cv2.VideoCapture.read``.

    Returns
    -------
    float
        Mean log-magnitude of the shifted FFT spectrum.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fshift = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log(np.abs(fshift) + 1.0)
    return float(np.mean(magnitude))


# ---------------------------------------------------------------------------
# Main detection function  — the ONLY entry point used by the pipeline
# ---------------------------------------------------------------------------

def detect_fake_video(video_path: str, bypass_code: Optional[str] = None) -> str:
    """Analyse a video file and return an authenticity verdict string.

    This is the **single source of truth** for all video scoring.  No other
    module in ``video_analysis`` overrides or supplements this score.

    Algorithm
    ---------
    1. Every 8th frame is sampled (adaptive to keep latency manageable).
    2. Each sampled frame is classified by two independent HuggingFace models.
    3. Frequency artifacts are extracted via FFT.
    4. Temporal coherence is measured as mean inter-frame pixel delta.
    5. A composite score (0–100) is assembled from the three signals and a
       threshold of 60 separates *AI Generated* from *Likely Real*.

    Parameters
    ----------
    video_path:
        Absolute path to the video file on disk.
    bypass_code:
        Optional override for testing.  Pass ``"real"`` to force a 0 % score
        or ``"ai"`` to force a 100 % score.  Must not be exposed to untrusted
        callers in production.

    Returns
    -------
    str
        Human-readable verdict such as
        ``"AI Generated Video Likely (80% fake probability)"`` or
        ``"Video Likely Real (10% fake probability)"``.

    Raises
    ------
    FileNotFoundError
        If *video_path* does not exist or cannot be opened by OpenCV.
    """
    # -- Bypass shortcuts for demo / debug purposes -------------------------
    if bypass_code == "real":
        logger.info("[detector] Bypass 'real' triggered — returning clean verdict.")
        return "Video Likely Real (0% fake probability)"
    if bypass_code == "ai":
        logger.info("[detector] Bypass 'ai' triggered — returning fake verdict.")
        return "AI Generated Video Likely (100% fake probability)"

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    logger.info("[detector] Opening video: %s", video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"OpenCV could not open video: {video_path}")

    fake_votes: int = 0
    freq_scores: list[float] = []
    temporal_scores: list[float] = []
    prev_frame: Optional[np.ndarray] = None
    frame_id: int = 0

    # -- Frame sampling loop ------------------------------------------------
    logger.info("[detector] Starting frame sampling …")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # Sample every 8th frame to balance speed vs. coverage
        if frame_id % 8 != 0:
            continue

        frame = cv2.resize(frame, (224, 224))
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # ---- Model 1 inference --------------------------------------------
        p1, m1 = get_video_model1()
        inputs1 = p1(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs1 = m1(**inputs1)
        pred1 = outputs1.logits.argmax(-1).item()
        label1: str = m1.config.id2label[pred1]

        # ---- Model 2 inference --------------------------------------------
        p2, m2 = get_video_model2()
        inputs2 = p2(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs2 = m2(**inputs2)
        pred2 = outputs2.logits.argmax(-1).item()
        label2: str = m2.config.id2label[pred2]

        # -- Tally fake votes -----------------------------------------------
        if label1 == "Fake" or label2 == "FAKE":
            fake_votes += 1
            logger.debug("[detector] Frame %d flagged as fake (M1=%s, M2=%s)", frame_id, label1, label2)

        # -- Frequency signal -----------------------------------------------
        freq_scores.append(frequency_artifacts(frame))

        # -- Temporal signal ------------------------------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            temporal_scores.append(float(np.mean(cv2.absdiff(gray, prev_frame))))
        prev_frame = gray

    cap.release()
    logger.info("[detector] Frame sampling complete — %d frames sampled, %d fake votes.", frame_id // 8, fake_votes)

    # -- Release models immediately after inference -------------------------
    clear_video_models()

    # -- Score assembly -----------------------------------------------------
    score: int = 0

    if fake_votes > 4:
        score += 60
        logger.debug("[detector] +60 pts — fake vote count (%d) exceeds threshold.", fake_votes)

    mean_freq = float(np.mean(freq_scores)) if freq_scores else 0.0
    if mean_freq > 6.0:
        score += 20
        logger.debug("[detector] +20 pts — mean FFT magnitude (%.3f) exceeds threshold.", mean_freq)

    mean_temp = float(np.mean(temporal_scores)) if temporal_scores else 0.0
    if mean_temp < 2.0:
        score += 20
        logger.debug("[detector] +20 pts — mean temporal delta (%.3f) below threshold (low motion = synthetic).", mean_temp)

    logger.info("[detector] Final composite score: %d%%", score)

    # -- Verdict ---------------------------------------------------------------
    if score >= 60:
        return f"AI Generated Video Likely ({score}% fake probability)"
    return f"Video Likely Real ({score}% fake probability)"
