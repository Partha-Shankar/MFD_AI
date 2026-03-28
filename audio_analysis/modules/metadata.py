"""
metadata.py — Phase 0: Container & Format Forensics

Forensic basis: Authentic recordings carry consistent metadata signatures from
real recording chains (microphone → ADC → encoder). Synthetic audio often has
inconsistent or absent metadata: no encoder tag, mismatched sample rates,
file size anomalies, and encoder strings that reveal TTS/voice-synthesis tools.

This module detects these inconsistencies without any audio signal processing.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional

import soundfile as sf
import mutagen

logger = logging.getLogger(__name__)

# Magic bytes for format detection
FORMAT_MAGIC: dict[bytes, str] = {
    b"\x49\x44\x33": "MP3",
    b"\xff\xfb": "MP3",
    b"\xff\xf3": "MP3",
    b"\xff\xf2": "MP3",
    b"RIFF": "WAV",
    b"fLaC": "FLAC",
    b"OggS": "OGG",
    b"free": "MP4",
    b"ftyp": "MP4",
    b"\x00\x00\x00\x1c": "MP4",
}

EXTENSION_MAP: dict[str, str] = {
    ".mp3": "MP3",
    ".wav": "WAV",
    ".flac": "FLAC",
    ".ogg": "OGG",
    ".m4a": "MP4",
    ".aac": "AAC",
    ".opus": "OGG",
}

STANDARD_SAMPLE_RATES: set[int] = {8000, 16000, 22050, 24000, 44100, 48000}

SYNTHETIC_ENCODER_KEYWORDS: list[str] = [
    # Classic TTS
    "festival", "espeak", "tacotron", "wavenet", "fastspeech",
    # Modern neural TTS
    "bark", "coqui", "tortoise", "vits", "elevenlabs", "resemble",
    "tts", "synthesizer", "text-to-speech", "glow-tts", "hifi-gan",
    "voicebox", "xtts", "yourtts",
    # Voice cloning platforms
    "playht", "murf", "speechify", "replica", "descript", "aiva",
    "resemble.ai", "uberduck", "wellsaid", "naturalreaders",
    # TTS framework encoders
    "silero", "pyttsx", "openvoice", "vall-e", "voicecraft",
    "styletts", "matcha", "vocos", "bigvgan",
]

# Codec compression ratios for size estimation
CODEC_RATIO: dict[str, float] = {
    "MP3": 0.09,
    "AAC": 0.11,
    "OGG": 0.10,
    "FLAC": 0.60,
    "WAV": 1.00,
    "MP4": 0.11,
}


class MetadataAnalyzer:
    """
    Analyzes audio file metadata for forensic anomalies.

    Checks format consistency, sample rate validity, encoder signatures,
    bitrate patterns, file size consistency, and timestamp anomalies.
    Returns a suspicion score and list of forensic flags.
    """

    def analyze(self, filepath: str) -> dict:
        """
        Perform full metadata forensic analysis on an audio file.

        Args:
            filepath: Path to the audio file.

        Returns:
            dict with metadata_score (0=clean, 1=suspicious), flags, and details.
        """
        start_time = time.time()
        flags: list[str] = []
        details: dict = {}

        filepath = str(filepath)

        try:
            # --- Basic file stats ---
            file_size = os.path.getsize(filepath)
            file_mtime = os.path.getmtime(filepath)
            details["file_size_bytes"] = file_size

            # --- 1. Format mismatch detection ---
            detected_format = self._detect_format_magic(filepath)
            ext = Path(filepath).suffix.lower()
            expected_format = EXTENSION_MAP.get(ext, "UNKNOWN")
            details["format"] = detected_format or expected_format

            if detected_format and expected_format != "UNKNOWN":
                if detected_format != expected_format:
                    flags.append("format_extension_mismatch")
                    logger.debug(f"Format mismatch: detected={detected_format}, ext={expected_format}")

            # --- 2. Sample rate suspicion ---
            try:
                info = sf.info(filepath)
                sr = info.samplerate
                channels = info.channels
                duration = info.duration
                details["sample_rate"] = sr
                details["channels"] = channels
                details["duration"] = duration

                if sr not in STANDARD_SAMPLE_RATES:
                    flags.append("suspicious_sample_rate")
                    logger.debug(f"Non-standard sample rate: {sr}")
                elif sr == 22050 and file_size > 1_000_000:
                    # 22050 Hz is the canonical TTS rate — flag if large file claims it
                    flags.append("suspicious_sample_rate")
                    logger.debug("22050 Hz sample rate on large file — common TTS indicator")

            except Exception as e:
                logger.warning(f"soundfile.info failed: {e}")
                sr, channels, duration = 44100, 1, 0.0
                details["sample_rate"] = sr
                details["channels"] = channels
                details["duration"] = duration

            # --- 3. Bitrate analysis (MP3/AAC/OGG) ---
            bitrate = None
            encoder = None
            try:
                audio_meta = mutagen.File(filepath)
                if audio_meta is not None:
                    # Extract bitrate
                    if hasattr(audio_meta, "info") and hasattr(audio_meta.info, "bitrate"):
                        bitrate = audio_meta.info.bitrate
                        details["bitrate"] = bitrate
                        if bitrate is not None and bitrate < 64000:
                            flags.append("suspiciously_low_bitrate")

                    # Check VBR/CBR consistency for MP3
                    if hasattr(audio_meta.info, "bitrate_mode"):
                        mode = str(audio_meta.info.bitrate_mode)
                        if "CBR" in mode.upper() and bitrate:
                            # CBR claims fixed bitrate — we can't verify variance here
                            # but flag as suspicious if bitrate is very round (exact 128/192/320)
                            pass  # Conservative — skip VBR/CBR mismatch without full stream scan

                    # Extract encoder signature
                    tag_keys_to_check = [
                        "encoder", "encoded_by", "encoding", "software",
                        "TSSE", "TENC", "comment", "\xa9too",
                    ]
                    for key in tag_keys_to_check:
                        val = audio_meta.get(key)
                        if val:
                            encoder = str(val[0]) if isinstance(val, list) else str(val)
                            break

                    details["encoder"] = encoder

                    if encoder:
                        enc_lower = encoder.lower()
                        if any(kw in enc_lower for kw in SYNTHETIC_ENCODER_KEYWORDS):
                            flags.append("synthetic_encoder_detected")
                            logger.debug(f"Synthetic encoder detected: {encoder}")
                        else:
                            # Only flag unknown if there's genuinely no encoder info
                            # Lavf/LAME/ffmpeg are neutral encoders — not suspicious in isolation
                            neutral_encoders = ["lavf", "lame", "ffmpeg", "libav", "fdk"]
                            if not any(ne in enc_lower for ne in neutral_encoders):
                                flags.append("unknown_encoder")

                else:
                    details["encoder"] = None
                    flags.append("unknown_encoder")

            except Exception as e:
                logger.warning(f"mutagen analysis failed: {e}")
                details["bitrate"] = None
                details["encoder"] = None
                flags.append("unknown_encoder")

            # --- 4. Duration vs file size consistency ---
            if duration > 0:
                fmt = details.get("format", "WAV")
                codec_ratio = CODEC_RATIO.get(fmt, 0.10)
                bit_depth = 16  # assume 16-bit
                expected_bytes_raw = sr * channels * (bit_depth / 8) * duration
                expected_bytes = expected_bytes_raw * codec_ratio
                # Allow 30% tolerance
                if expected_bytes > 0:
                    size_ratio = file_size / expected_bytes
                    if size_ratio < 0.7 or size_ratio > 1.3:
                        flags.append("size_duration_mismatch")
                        logger.debug(
                            f"Size mismatch: actual={file_size}, expected≈{expected_bytes:.0f}, "
                            f"ratio={size_ratio:.2f}"
                        )

            # --- 5. Modification time anomaly ---
            try:
                audio_meta2 = mutagen.File(filepath)
                creation_ts = None
                if audio_meta2:
                    for key in ["creation_time", "date", "TDRC", "TDOR", "\xa9day"]:
                        val = audio_meta2.get(key)
                        if val:
                            creation_ts = str(val[0]) if isinstance(val, list) else str(val)
                            break
                if creation_ts:
                    # Try to parse creation timestamp
                    import datetime
                    try:
                        # Handle various formats
                        for fmt_str in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%Y"]:
                            try:
                                ct = datetime.datetime.strptime(creation_ts[:len(fmt_str)], fmt_str)
                                ct_ts = ct.timestamp()
                                if abs(file_mtime - ct_ts) > 3600:  # >1 hour difference
                                    flags.append("timestamp_inconsistency")
                                break
                            except ValueError:
                                continue
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"Timestamp check failed: {e}")

        except Exception as e:
            logger.error(f"MetadataAnalyzer error: {e}")
            details.setdefault("format", "UNKNOWN")
            details.setdefault("sample_rate", 0)
            details.setdefault("channels", 0)
            details.setdefault("duration", 0.0)
            details.setdefault("bitrate", None)
            details.setdefault("encoder", None)
            flags.append("analysis_error")

        # --- Score calculation ---
        metadata_score = min(1.0, len(flags) * 0.25)

        return {
            "metadata_score": round(metadata_score, 4),
            "flags": flags,
            "details": details,
            "analysis_time_seconds": round(time.time() - start_time, 3),
        }

    def _detect_format_magic(self, filepath: str) -> Optional[str]:
        """Read first 12 bytes and match against known magic byte signatures."""
        try:
            with open(filepath, "rb") as f:
                header = f.read(12)
            for magic, fmt in FORMAT_MAGIC.items():
                if header[:len(magic)] == magic:
                    return fmt
            # Check RIFF WAV specifically (bytes 8-11 = "WAVE")
            if header[:4] == b"RIFF" and header[8:12] == b"WAVE":
                return "WAV"
        except Exception as e:
            logger.warning(f"Magic byte detection failed: {e}")
        return None
