"""
video_analysis/progress_messages.py
=====================================
Localised, stage-aware progress message generator for video forensics.

Provides human-readable status messages that correspond to each phase of the
video analysis pipeline.  These messages are surfaced through the system
logging infrastructure and, optionally, streamed to the frontend via
Server-Sent Events (SSE) for real-time UI feedback.

The message catalogue is grouped by forensic stage so the frontend
``AnalysisLogs`` component can display contextually appropriate agent labels
and log lines as each phase activates.

Standalone module — not wired into the active pipeline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterator, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage enumeration
# ---------------------------------------------------------------------------

class VideoStage(str, Enum):
    """Ordered stages of the video forensics pipeline.

    Values correspond to the ``agent`` labels rendered in the frontend
    ``AnalysisLogs`` component.
    """

    INIT               = "System Core"
    ALLOCATE           = "Orchestrator"
    PARSE              = "Media Parser"
    FRAME_SAMPLE       = "Frame Engine"
    VISION_AI          = "Vision AI"
    FRAME_ANALYZE      = "Frame Analyzer"
    GAN_DETECT         = "GAN Detector"
    TEXTURE            = "Texture Engine"
    EDGE               = "Edge Inspector"
    TEMPORAL           = "Temporal Agent"
    MOTION             = "Motion Tracker"
    DYNAMICS           = "Dynamics Engine"
    POSE               = "Pose Analyzer"
    INTERPOLATION      = "Interpolation Detector"
    SIGNAL             = "Signal Processor"
    FFT                = "FFT Analyzer"
    NOISE              = "Noise Model"
    COMPRESSION        = "Compression Inspector"
    SEMANTIC           = "Semantic AI"
    CONTEXT            = "Context Engine"
    PHYSICS            = "Physics Validator"
    REALITY            = "Reality Checker"
    BIOMETRIC          = "Biometric Agent"
    MICRO_EXPRESSION   = "Micro-Expression Engine"
    DEPTH              = "Depth Analyzer"
    REFLECTION         = "Reflection Inspector"
    PHOTOMETRIC        = "Photometric Engine"
    CONSENSUS          = "Consensus Engine"
    CONFIDENCE         = "Confidence Model"
    ANOMALY            = "Anomaly Detector"
    DECISION           = "Decision Engine"
    REPORT             = "Report Generator"
    EXPLAINABILITY     = "Explainability Engine"
    UI_FORMAT          = "UI Formatter"
    COMPLETE           = "System Core"


# ---------------------------------------------------------------------------
# Message catalogue
# ---------------------------------------------------------------------------

# Mapping of stage → ordered list of candidate log messages.
# The frontend cycles through these as the stage progresses.
_STAGE_MESSAGES: dict[VideoStage, List[str]] = {
    VideoStage.INIT: [
        "Initializing multi-agent forensic pipeline…",
        "Warming up inference engines…",
        "Verifying CUDA availability for accelerated inference…",
    ],
    VideoStage.ALLOCATE: [
        "Allocating resources for video analysis…",
        "Reserving video decode buffer (512 MB)…",
        "Spawning worker threads for parallel frame processing…",
    ],
    VideoStage.PARSE: [
        "Extracting metadata and decoding video stream…",
        "Parsing container headers (MP4 / AVI / MOV)…",
        "Identifying codec, bitrate, and frame rate…",
        "Validating stream integrity checksums…",
    ],
    VideoStage.FRAME_SAMPLE: [
        "Sampling keyframes at adaptive intervals…",
        "Applying scene-change detection to refine sample set…",
        "Decoded %d keyframes for analysis…",
    ],
    VideoStage.VISION_AI: [
        "Running frame-level authenticity classification…",
        "Forwarding frame batch to deepfake-detector-model-v1…",
        "Forwarding frame batch to deepfake_vs_real_image_detection…",
        "Aggregating dual-model ensemble votes…",
    ],
    VideoStage.FRAME_ANALYZE: [
        "Detecting spatial inconsistencies in frame batch…",
        "Computing per-patch anomaly scores…",
        "Flagging %d suspicious regions in sampled frames…",
    ],
    VideoStage.GAN_DETECT: [
        "Scanning for generative model fingerprints…",
        "Checking for StyleGAN / Stable Diffusion upsampling artifacts…",
        "Evaluating frequency comb patterns characteristic of GAN generators…",
    ],
    VideoStage.TEXTURE: [
        "Evaluating surface-level texture realism…",
        "Comparing local binary pattern histograms to reference distribution…",
        "Measuring skin texture micro-grain variance…",
    ],
    VideoStage.EDGE: [
        "Checking structural boundary coherence…",
        "Applying Canny edge-detection at three scales…",
        "Computing edge gradient magnitude histogram…",
    ],
    VideoStage.TEMPORAL: [
        "Analyzing frame-to-frame transitions…",
        "Measuring inter-frame pixel delta (temporal coherence)…",
        "Checking for synthetic frame blending artifacts…",
    ],
    VideoStage.MOTION: [
        "Computing optical flow consistency…",
        "Running Lucas-Kanade optical flow on feature points…",
        "Detecting unphysical motion vectors…",
    ],
    VideoStage.DYNAMICS: [
        "Evaluating motion smoothness and jitter patterns…",
        "Fitting natural motion priors to detected trajectories…",
        "Computing jerk and acceleration plausibility scores…",
    ],
    VideoStage.POSE: [
        "Tracking head pose stability across frames…",
        "Estimating pitch / yaw / roll from facial landmarks…",
        "Detecting discontinuous pose transitions…",
    ],
    VideoStage.INTERPOLATION: [
        "Checking for synthetic frame generation…",
        "Evaluating RIFE / DAIN interpolation artifact signatures…",
        "Measuring inter-frame content similarity against interpolation profile…",
    ],
    VideoStage.SIGNAL: [
        "Performing frequency domain decomposition…",
        "Computing 2-D FFT for each sampled keyframe…",
        "Accumulating spectral energy histogram…",
    ],
    VideoStage.FFT: [
        "Detecting spectral anomalies…",
        "Locating periodic peaks in FFT magnitude spectrum…",
        "Comparing to authentic camera frequency profiles…",
    ],
    VideoStage.NOISE: [
        "Comparing sensor noise distribution…",
        "Extracting PRNU noise residual via wavelet denoising…",
        "Computing inter-frame PRNU correlation…",
    ],
    VideoStage.COMPRESSION: [
        "Analyzing encoding artifacts…",
        "Running ELA (Error Level Analysis) on keyframes…",
        "Measuring DCT coefficient distribution…",
    ],
    VideoStage.SEMANTIC: [
        "Validating scene coherence…",
        "Running CLIP semantic alignment check…",
        "Measuring cross-modal scene-text consistency…",
    ],
    VideoStage.CONTEXT: [
        "Checking object relationships…",
        "Running object detector to verify scene plausibility…",
        "Flagging objects that violate scene context rules…",
    ],
    VideoStage.PHYSICS: [
        "Evaluating lighting and shadow consistency…",
        "Estimating light source direction from facial reflections…",
        "Checking shadow geometry against inferred illumination model…",
    ],
    VideoStage.REALITY: [
        "Detecting logical inconsistencies…",
        "Checking facial geometry against 3DMM priors…",
        "Validating ear / eye / nose symmetry constraints…",
    ],
    VideoStage.BIOMETRIC: [
        "Evaluating blink patterns and facial dynamics…",
        "Measuring inter-blink interval distribution…",
        "Checking microsaccade frequency against human baseline…",
    ],
    VideoStage.MICRO_EXPRESSION: [
        "Analyzing subtle emotional variations…",
        "Detecting Action Unit (AU) consistency across frames…",
        "Checking for unnatural AU co-occurrence patterns…",
    ],
    VideoStage.DEPTH: [
        "Estimating scene depth consistency…",
        "Running monocular depth estimator on keyframes…",
        "Checking depth discontinuities at face boundaries…",
    ],
    VideoStage.REFLECTION: [
        "Checking eye and surface reflections…",
        "Extracting corneal reflection maps from high-res frames…",
        "Comparing left / right eye corneal reflections for consistency…",
    ],
    VideoStage.PHOTOMETRIC: [
        "Validating illumination uniformity…",
        "Fitting Lambertian shading model to detected face geometry…",
        "Computing residual shading error across the face region…",
    ],
    VideoStage.CONSENSUS: [
        "Aggregating multi-layer forensic signals…",
        "Weighting 14 independent signals by calibrated confidence…",
        "Computing weighted composite authenticity score…",
    ],
    VideoStage.CONFIDENCE: [
        "Resolving conflicting indicators…",
        "Applying Bayesian belief propagation across signal graph…",
        "Computing posterior probability of AI synthesis…",
    ],
    VideoStage.ANOMALY: [
        "Ranking detected irregularities…",
        "Sorting forensic flags by weighted risk contribution…",
        "Identifying top-3 most discriminative features…",
    ],
    VideoStage.DECISION: [
        "Computing final authenticity score…",
        "Applying decision threshold at 60% composite fake probability…",
        "Finalising verdict with confidence bounds…",
    ],
    VideoStage.REPORT: [
        "Constructing explainable forensic report…",
        "Serialising per-signal contributions to JSON payload…",
        "Attaching ELA visualisation overlay…",
    ],
    VideoStage.EXPLAINABILITY: [
        "Translating signals into human-readable insights…",
        "Generating natural language explanation from top-3 features…",
        "Attaching recommended next steps for the analyst…",
    ],
    VideoStage.UI_FORMAT: [
        "Preparing visualization payload…",
        "Encoding ELA heatmap as base64 PNG…",
        "Serializing response for frontend consumption…",
    ],
    VideoStage.COMPLETE: [
        "Analysis complete.",
        "Pipeline teardown — releasing model weights from memory…",
        "Temporary files cleaned up.",
    ],
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProgressMessage:
    """A single progress log message.

    Attributes
    ----------
    stage:
        The forensic stage this message belongs to.
    agent:
        Display name of the agent (matches frontend ``Stage.agent``).
    message:
        Human-readable log line.
    timestamp:
        Unix epoch float at the time of creation.
    """

    stage: VideoStage
    agent: str
    message: str
    timestamp: float


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def iter_progress(
    stages: Optional[List[VideoStage]] = None,
    callback: Optional[Callable[[ProgressMessage], None]] = None,
) -> Iterator[ProgressMessage]:
    """Iterate over progress messages for a sequence of video analysis stages.

    Yields one ``ProgressMessage`` per stage (using the first message in the
    catalogue).  Optionally invokes *callback* for each message so callers can
    push to an SSE queue or log stream without collecting the full list.

    Parameters
    ----------
    stages:
        Ordered list of stages to emit.  Defaults to all stages in order.
    callback:
        Optional callable invoked with each ``ProgressMessage`` immediately
        after it is yielded.

    Yields
    ------
    ProgressMessage
    """
    if stages is None:
        stages = list(VideoStage)

    for stage in stages:
        messages = _STAGE_MESSAGES.get(stage, [stage.value])
        msg = ProgressMessage(
            stage=stage,
            agent=stage.value,
            message=messages[0],
            timestamp=time.time(),
        )
        logger.debug("[progress] [%s] %s", msg.agent, msg.message)
        if callback is not None:
            callback(msg)
        yield msg


def get_stage_messages(stage: VideoStage) -> List[str]:
    """Return all catalogue messages for a given forensic stage.

    Parameters
    ----------
    stage:
        The forensic stage to query.

    Returns
    -------
    list of str
        All candidate log messages for the stage.
    """
    return list(_STAGE_MESSAGES.get(stage, [stage.value]))


def as_frontend_stages() -> List[dict]:
    """Return the full stage catalogue in the format expected by the frontend.

    The frontend ``AnalysisLogs`` component expects a list of objects with
    ``agent`` and ``message`` keys.

    Returns
    -------
    list of dict
        Each element has keys ``"agent"`` and ``"message"``.
    """
    result = []
    for stage in VideoStage:
        messages = _STAGE_MESSAGES.get(stage, [stage.value])
        result.append({"agent": stage.value, "message": messages[0]})
    return result
