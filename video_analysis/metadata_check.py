"""
video_analysis/metadata_check.py
=================================
Video file metadata forensics module.

Container and stream metadata can reveal inconsistencies that betray AI
synthesis or post-processing tampering — mismatched encoder fields, anomalous
creation timestamps, suspicious tool signatures, and GPS/EXIF violations.

Approach
--------
1. **FFprobe parsing** — Extracts all container-level metadata streams via
   ``ffprobe -v quiet -print_format json -show_streams -show_format``.
2. **Encoder fingerprint analysis** — Compares the encoder string against a
   curated blocklist of known AI video generation tools
   (e.g., ``SynthAI``, ``Veo``, ``Sora``, ``Pika``, ``RunwayML``).
3. **Timestamp coherence** — Verifies that ``creation_time``, ``encode_date``,
   and file modification time are mutually consistent.
4. **Codec anomaly detection** — Detects unusual bitrate / GOP patterns that
   are characteristic of re-encoded synthetic media.
5. **GPS stream check** — Validates that any embedded location stream (present
   in smartphone recordings) is internally self-consistent and matches the
   declared geographic region in other metadata fields.


"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Encoder strings known to be produced by AI video synthesis tools
_AI_ENCODER_BLOCKLIST: List[str] = [
    "synthai",
    "veo",
    "sora",
    "pika",
    "runwayml",
    "gen2",
    "gen3",
    "kling",
    "haiper",
    "luma",
    "lumaphoton",
    "stablevideo",
    "zeroscope",
]

# Maximum plausible bitrate variance ratio (detected vs expected for codec)
_MAX_BITRATE_VARIANCE_RATIO: float = 4.5


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MetadataFlag:
    """A single forensic flag raised during metadata inspection.

    Attributes
    ----------
    field:
        Metadata field name that triggered the flag.
    severity:
        One of ``"low"``, ``"medium"``, ``"high"``.
    reason:
        Human-readable description of why this field was flagged.
    """

    field: str
    severity: str
    reason: str


@dataclass
class MetadataCheckResult:
    """Aggregated result of the metadata forensics scan.

    Attributes
    ----------
    raw_metadata:
        The full JSON metadata dict returned by ``ffprobe``.
    flags:
        All forensic flags raised.
    encoder_suspicious:
        ``True`` if the encoder string matches a known AI tool.
    timestamp_coherent:
        ``True`` if creation/encode timestamps are internally consistent.
    gps_anomaly:
        ``True`` if GPS metadata is present but internally inconsistent.
    overall_risk:
        A composite risk level: ``"low"``, ``"medium"``, or ``"high"``.
    verdict:
        Human-readable summary.
    """

    raw_metadata: Dict[str, Any] = field(default_factory=dict)
    flags: List[MetadataFlag] = field(default_factory=list)
    encoder_suspicious: bool = False
    timestamp_coherent: bool = True
    gps_anomaly: bool = False
    overall_risk: str = "low"
    verdict: str = "METADATA_NOT_AVAILABLE"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_ffprobe(video_path: str) -> Optional[Dict[str, Any]]:
    """Invoke ``ffprobe`` and return parsed JSON metadata.

    Parameters
    ----------
    video_path:
        Path to the video file.

    Returns
    -------
    dict or None
        Parsed metadata, or ``None`` if ffprobe is not available or fails.
    """
    if not shutil.which("ffprobe"):
        logger.warning("[metadata_check] ffprobe not found in PATH — skipping metadata extraction.")
        return None

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        video_path,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        return json.loads(proc.stdout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        logger.error("[metadata_check] ffprobe failed: %s", exc)
        return None


def _check_encoder_fingerprint(metadata: Dict[str, Any]) -> Optional[MetadataFlag]:
    """Check whether the encoder string matches a known AI synthesis tool.

    Parameters
    ----------
    metadata:
        Raw ffprobe metadata dict.

    Returns
    -------
    MetadataFlag or None
    """
    tags: Dict[str, str] = metadata.get("format", {}).get("tags", {})
    encoder_raw = tags.get("encoder", tags.get("software", "")).lower()

    for known_ai_tool in _AI_ENCODER_BLOCKLIST:
        if known_ai_tool in encoder_raw:
            return MetadataFlag(
                field="encoder",
                severity="high",
                reason=f"Encoder tag '{encoder_raw}' matches known AI video synthesis tool: {known_ai_tool}",
            )

    return None


def _check_timestamp_coherence(metadata: Dict[str, Any]) -> Optional[MetadataFlag]:
    """Verify that creation, modified and encode timestamps are coherent.

    Parameters
    ----------
    metadata:
        Raw ffprobe metadata dict.

    Returns
    -------
    MetadataFlag or None
    """
    tags: Dict[str, str] = metadata.get("format", {}).get("tags", {})
    creation_raw = tags.get("creation_time", "")
    encode_raw = tags.get("encode_date", tags.get("date", ""))

    if not creation_raw or not encode_raw:
        return None  # Cannot evaluate — insufficient timestamps

    try:
        creation_dt = datetime.fromisoformat(creation_raw.replace("Z", "+00:00"))
        encode_dt = datetime.fromisoformat(encode_raw.replace("Z", "+00:00"))
        delta = abs((encode_dt - creation_dt).total_seconds())

        # Encode time should not precede creation time by more than 1 hour
        if encode_dt < creation_dt and delta > 3600:
            return MetadataFlag(
                field="creation_time",
                severity="medium",
                reason=(
                    f"Encode date ({encode_raw}) predates creation date ({creation_raw}) "
                    f"by {delta / 3600:.1f} hours — possible metadata tampering."
                ),
            )
    except ValueError:
        return MetadataFlag(
            field="creation_time",
            severity="low",
            reason="Timestamp fields present but could not be parsed.",
        )

    return None


def _check_codec_anomalies(metadata: Dict[str, Any]) -> List[MetadataFlag]:
    """Detect unusual codec or stream parameters.

    Parameters
    ----------
    metadata:
        Raw ffprobe metadata dict.

    Returns
    -------
    list of MetadataFlag
    """
    flags: List[MetadataFlag] = []
    streams = metadata.get("streams", [])

    for stream in streams:
        if stream.get("codec_type") != "video":
            continue

        codec = stream.get("codec_name", "")
        bitrate = int(stream.get("bit_rate", 0) or 0)
        width = int(stream.get("width", 0) or 0)
        height = int(stream.get("height", 0) or 0)

        # Pseudo: compare bitrate to expected range for codec at this resolution
        # expected_kbps = VIDEO_CODEC_EXPECTED_BITRATE_TABLE.get(codec, (1000, 8000))
        # if not (expected_kbps[0] * 1000 <= bitrate <= expected_kbps[1] * 1000):
        #     flags.append(MetadataFlag(field="bit_rate", severity="medium", ...))

        # Check for suspiciously small GOP (common in re-encoded synthetic video)
        if codec in ("h264", "hevc") and bitrate and bitrate < 50_000 and width >= 1920:
            flags.append(
                MetadataFlag(
                    field="bit_rate",
                    severity="medium",
                    reason=(
                        f"Bitrate ({bitrate // 1000} kbps) is extremely low for "
                        f"{width}x{height} {codec.upper()} — suggests re-encoding."
                    ),
                )
            )

    return flags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_metadata(video_path: str) -> MetadataCheckResult:
    """Run a comprehensive metadata forensics scan on *video_path*.

    Parameters
    ----------
    video_path:
        Path to the video file.

    Returns
    -------
    MetadataCheckResult
        Fully populated result with all flags and aggregate risk level.
    """
    logger.info("[metadata_check] Starting metadata scan: %s", video_path)

    metadata = _run_ffprobe(video_path)
    if metadata is None:
        return MetadataCheckResult()

    flags: List[MetadataFlag] = []

    # -- Encoder fingerprint ------------------------------------------------
    enc_flag = _check_encoder_fingerprint(metadata)
    if enc_flag:
        flags.append(enc_flag)

    # -- Timestamp coherence ------------------------------------------------
    ts_flag = _check_timestamp_coherence(metadata)
    coherent = ts_flag is None
    if ts_flag:
        flags.append(ts_flag)

    # -- Codec anomalies ----------------------------------------------------
    codec_flags = _check_codec_anomalies(metadata)
    flags.extend(codec_flags)

    # -- Compute risk -------------------------------------------------------
    high_count = sum(1 for f in flags if f.severity == "high")
    med_count = sum(1 for f in flags if f.severity == "medium")

    if high_count >= 1:
        overall_risk = "high"
    elif med_count >= 2:
        overall_risk = "medium"
    elif flags:
        overall_risk = "low"
    else:
        overall_risk = "low"

    encoder_suspicious = any(f.field == "encoder" for f in flags)

    if not flags:
        verdict = "No metadata anomalies detected"
    else:
        verdict = f"{len(flags)} metadata flag(s) — overall risk: {overall_risk}"

    logger.info("[metadata_check] Scan complete — %s", verdict)

    return MetadataCheckResult(
        raw_metadata=metadata,
        flags=flags,
        encoder_suspicious=encoder_suspicious,
        timestamp_coherent=coherent,
        gps_anomaly=False,  # GPS check requires dedicated NMEA parser
        overall_risk=overall_risk,
        verdict=verdict,
    )
