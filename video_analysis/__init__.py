"""
video_analysis
==============
Public surface of the video analysis package.

Only ``detect_fake_video`` is exported.  All other modules in this package
are internal and must not be imported by callers outside the package.
"""

from video_analysis.detector import detect_fake_video  # noqa: F401

__all__ = ["detect_fake_video"]
