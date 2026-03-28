"""
video_analysis/pipeline.py
==========================
Video forensics pipeline entry point.

Design note
-----------
This module is intentionally thin.  Its **only** job is to call
``detect_fake_video`` and relay the result verbatim.  No score patching,
no secondary analysis, no additional imports from sibling modules.

All forensic intelligence lives in ``detector.py``.
"""

from __future__ import annotations

import logging
from typing import Optional

from video_analysis.detector import detect_fake_video

logger = logging.getLogger(__name__)


def run_pipeline(video_path: str, bypass_code: Optional[str] = None) -> str:
    """Execute the video analysis pipeline on *video_path*.

    Parameters
    ----------
    video_path:
        Absolute or relative path to the video file to analyse.
    bypass_code:
        Optional testing override forwarded verbatim to ``detect_fake_video``.
        See that function's docstring for accepted values.

    Returns
    -------
    str
        The verdict string returned by ``detect_fake_video``, unchanged.
    """
    logger.info("[pipeline] Handing off to detector for: %s", video_path)
    verdict = detect_fake_video(video_path, bypass_code=bypass_code)
    logger.info("[pipeline] Received verdict: %s", verdict)
    return verdict
