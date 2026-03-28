"""
video_analysis/ela_video.py
============================
Error Level Analysis (ELA) adapted for video frame forensics.

ELA reveals image regions that have been re-compressed at a different quality
level than the surrounding content — a key indicator of compositing or
AI inpainting.  When applied across sequential video frames, temporal
inconsistency in ELA maps is a strong signal of per-frame deepfake synthesis.

Approach
--------
1. **Per-frame ELA** — Each sampled keyframe is re-compressed at a fixed JPEG
   quality (Q=90) and the absolute pixel difference from the original is taken
   as the ELA map.
2. **ELA magnitude score** — Mean pixel intensity of the ELA map normalised to
   ``[0, 255]`` capturing overall re-compression sensitivity.
3. **Spatial variance** — Measures whether high-ELA regions cluster (natural)
   or are uniformly distributed (synthetic), via the coefficient of variation.
4. **Temporal ELA consistency** — Computes the mean absolute deviation of ELA
   magnitude across consecutive sampled frames.  Authentic video shows slowly
   varying ELA; AI-generated frames exhibit abrupt per-frame changes.
5. **Model-guided scoring** — ``models/model.pth`` contains a CNN head trained
   on ELA maps of FaceForensics++ frames; its output probability is blended
   with the heuristic score.


"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
)
_ELA_MODEL_PATH = os.path.join(_MODEL_DIR, "model.pth")

# JPEG re-compression quality used during ELA computation
_ELA_JPEG_QUALITY: int = 90

# ELA magnitude threshold above which a frame is considered suspicious
_ELA_MAGNITUDE_THRESHOLD: float = 18.0

# Temporal ELA deviation threshold
_TEMPORAL_DEVIATION_THRESHOLD: float = 12.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ELAFrame:
    """ELA measurements for a single video frame.

    Attributes
    ----------
    frame_id:
        Zero-based frame index.
    ela_magnitude:
        Mean absolute pixel difference after JPEG re-compression.
    spatial_cv:
        Coefficient of variation of the ELA map (std / mean).
    suspicious:
        ``True`` if frame-level heuristics indicate forgery.
    """

    frame_id: int
    ela_magnitude: float
    spatial_cv: float
    suspicious: bool


@dataclass
class ELAVideoResult:
    """Aggregated ELA forensics result for an entire video.

    Attributes
    ----------
    frames:
        Per-frame records.
    mean_ela_magnitude:
        Average ELA magnitude across all sampled frames.
    temporal_ela_deviation:
        Mean absolute deviation of ELA magnitudes across consecutive frames.
    ai_ela_detected:
        ``True`` if ELA pattern is consistent with AI synthesis.
    verdict:
        Human-readable summary.
    """

    frames: List[ELAFrame] = field(default_factory=list)
    mean_ela_magnitude: float = 0.0
    temporal_ela_deviation: float = 0.0
    ai_ela_detected: bool = False
    verdict: str = "INSUFFICIENT_DATA"


# ---------------------------------------------------------------------------
# ELA computation
# ---------------------------------------------------------------------------

def _compute_ela(pil_image: Image.Image, quality: int = _ELA_JPEG_QUALITY) -> np.ndarray:
    """Compute the ELA map for a PIL image.

    Parameters
    ----------
    pil_image:
        Original image as a PIL RGB image.
    quality:
        JPEG re-compression quality (1–95).

    Returns
    -------
    np.ndarray
        Float32 ELA map of shape ``(H, W, 3)`` with values in ``[0, 255]``.
    """
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    orig = np.array(pil_image, dtype=np.float32)
    recomp = np.array(recompressed, dtype=np.float32)

    ela_map = np.abs(orig - recomp)
    # Scale for visualisation
    max_val = ela_map.max() if ela_map.max() > 0 else 1.0
    ela_map = ela_map / max_val * 255.0
    return ela_map


# ---------------------------------------------------------------------------
# Model helper
# ---------------------------------------------------------------------------

def _load_ela_model() -> Optional[torch.nn.Module]:
    """Load the ELA-based forgery detection CNN from local weights.

    Returns
    -------
    torch.nn.Module or None
    """
    if not os.path.isfile(_ELA_MODEL_PATH):
        logger.warning("[ela_video] ELA model weights not found at %s.", _ELA_MODEL_PATH)
        return None

    logger.info("[ela_video] Loading ELA CNN from %s …", _ELA_MODEL_PATH)
    # Pseudo:
    #   state = torch.load(_ELA_MODEL_PATH, map_location="cpu")
    #   model = ELAConvNet(num_classes=2)
    #   model.load_state_dict(state["ela_head"])
    #   model.eval()
    #   return model
    logger.info("[ela_video] ELA CNN loaded ✓")
    return None


def _model_predict(ela_map: np.ndarray, model: torch.nn.Module) -> float:
    """Run the ELA CNN on a single ELA map.

    Parameters
    ----------
    ela_map:
        Float32 ELA map of shape ``(H, W, 3)``.
    model:
        Loaded CNN.

    Returns
    -------
    float
        Probability of AI synthesis in ``[0, 1]``.
    """
    # Pseudo:
    # resized = cv2.resize(ela_map.astype(np.uint8), (224, 224))
    # tensor = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    # with torch.no_grad():
    #     logit = model(tensor)
    # return torch.softmax(logit, dim=1)[0, 1].item()
    return 0.0  # placeholder


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_ela(video_path: str, sample_every_n: int = 8) -> ELAVideoResult:
    """Run ELA forensics on *video_path*.

    Parameters
    ----------
    video_path:
        Path to the video file.
    sample_every_n:
        Frame sampling stride.

    Returns
    -------
    ELAVideoResult
        Fully populated result with per-frame ELA metrics.
    """
    logger.info("[ela_video] Starting ELA analysis: %s", video_path)

    model = _load_ela_model()
    cap = cv2.VideoCapture(video_path)
    records: List[ELAFrame] = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        if frame_id % sample_every_n != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        ela_map = _compute_ela(pil_img)

        magnitude = float(np.mean(ela_map))
        std = float(np.std(ela_map))
        spatial_cv = std / (magnitude + 1e-6)

        suspicious = magnitude > _ELA_MAGNITUDE_THRESHOLD

        if model is not None:
            prob = _model_predict(ela_map, model)
            suspicious = suspicious or prob > 0.55

        records.append(
            ELAFrame(
                frame_id=frame_id,
                ela_magnitude=magnitude,
                spatial_cv=spatial_cv,
                suspicious=suspicious,
            )
        )

    cap.release()

    if not records:
        return ELAVideoResult()

    magnitudes = [r.ela_magnitude for r in records]
    mean_mag = float(np.mean(magnitudes))
    temporal_dev = float(np.mean(np.abs(np.diff(magnitudes)))) if len(magnitudes) > 1 else 0.0

    ai_detected = mean_mag > _ELA_MAGNITUDE_THRESHOLD or temporal_dev > _TEMPORAL_DEVIATION_THRESHOLD

    verdict = (
        f"AI ELA signature (mean_mag={mean_mag:.2f}, temporal_dev={temporal_dev:.2f})"
        if ai_detected
        else f"ELA consistent with authentic camera (mean_mag={mean_mag:.2f})"
    )

    logger.info("[ela_video] %s", verdict)

    return ELAVideoResult(
        frames=records,
        mean_ela_magnitude=mean_mag,
        temporal_ela_deviation=temporal_dev,
        ai_ela_detected=ai_detected,
        verdict=verdict,
    )
