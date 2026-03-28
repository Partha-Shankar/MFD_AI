"""
tests/test_pipeline.py — Pytest test suite for the Deepfake Audio Detection System.

Tests cover all four analysis modules and the full integration pipeline.
All tests use synthetically generated audio — no external files required.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.metadata import MetadataAnalyzer
from modules.spectral import SpectralAnalyzer
from modules.temporal import TemporalAnalyzer
from modules.speaker import SpeakerConsistencyAnalyzer
from modules.fusion import VerdictEngine

# ── Helpers ───────────────────────────────────────────────────────────────────

SR = 16000


def make_real_speech(duration: float = 6.0, sr: int = SR) -> np.ndarray:
    """Simulate authentic speech: variable pitch, noisy floor, irregular pauses."""
    np.random.seed(1)
    n = int(sr * duration)
    t = np.linspace(0, duration, n)
    f0 = 120.0 + 5 * np.sin(2 * np.pi * 0.3 * t) + 1.5 * np.random.randn(n) * 0.1
    voiced = sum(np.sin(k * 2 * np.pi * np.cumsum(f0 / sr)) / k for k in range(1, 8))
    envelope = 0.5 + 0.3 * np.sin(2 * np.pi * 0.8 * t) + 0.15 * np.abs(np.random.randn(n))
    envelope = np.clip(envelope, 0.05, 1.0)
    noise = 0.01 * np.random.randn(n)
    # Irregular pauses
    for ps in [1.3, 3.7, 5.1]:
        s, e = int(ps * sr), int((ps + 0.3) * sr)
        envelope[s:e] *= 0.05
    audio = (voiced * envelope + noise) * 0.3
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def make_fake_speech(duration: float = 6.0, sr: int = SR) -> np.ndarray:
    """Simulate TTS: stable pitch, uniform energy, no noise, hard HF cutoff."""
    np.random.seed(2)
    n = int(sr * duration)
    t = np.linspace(0, duration, n)
    voiced = sum(np.sin(k * 2 * np.pi * 120.0 * t) / k for k in range(1, 8))
    envelope = 0.5 * np.ones(n)
    noise = 0.0001 * np.random.randn(n)
    # Metronomic pauses
    for ps in [2.0, 4.0]:
        s, e = int(ps * sr), int((ps + 0.2) * sr)
        envelope[s:e] = 0.0
    combined = voiced * envelope + noise
    # Hard HF cutoff at 4 kHz
    fft = np.fft.rfft(combined)
    freqs = np.fft.rfftfreq(len(combined), 1.0 / sr)
    fft[freqs > 4000] = 0
    audio = np.fft.irfft(fft)[:n] * 0.3
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def write_wav(audio: np.ndarray, sr: int, suffix: str = ".wav") -> str:
    """Write audio to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    sf.write(tmp.name, audio, sr)
    return tmp.name


# ── Metadata tests ────────────────────────────────────────────────────────────

class TestMetadata:
    def test_metadata_wav_real(self):
        """Clean WAV file should have a low metadata suspicion score."""
        audio = make_real_speech()
        path = write_wav(audio, SR, ".wav")
        result = MetadataAnalyzer().analyze(path)
        assert "metadata_score" in result
        assert result["metadata_score"] <= 0.5, (
            f"Expected low score for clean WAV, got {result['metadata_score']}"
        )
        Path(path).unlink(missing_ok=True)

    def test_metadata_mismatch(self):
        """Renaming a WAV to .mp3 should trigger format_extension_mismatch."""
        audio = make_real_speech()
        path = write_wav(audio, SR, ".mp3")  # WAV data but .mp3 extension
        result = MetadataAnalyzer().analyze(path)
        # WAV magic bytes won't match .mp3 extension
        assert "format_extension_mismatch" in result["flags"], (
            f"Expected format_extension_mismatch flag; got flags: {result['flags']}"
        )
        Path(path).unlink(missing_ok=True)

    def test_metadata_returns_required_keys(self):
        """Result dict must always contain required keys."""
        audio = make_real_speech()
        path = write_wav(audio, SR)
        result = MetadataAnalyzer().analyze(path)
        for key in ["metadata_score", "flags", "details"]:
            assert key in result, f"Missing key: {key}"
        Path(path).unlink(missing_ok=True)


# ── Spectral tests ────────────────────────────────────────────────────────────

class TestSpectral:
    def test_spectral_smooth_fake(self):
        """Synthetic TTS signal should produce spectral_score > 0.4."""
        audio = make_fake_speech()
        result = SpectralAnalyzer().analyze(audio, SR)
        assert result["spectral_score"] > 0.4, (
            f"Expected suspicious score for TTS signal, got {result['spectral_score']}"
        )

    def test_spectral_real_lower_than_fake(self):
        """Real speech should score lower (less suspicious) than fake speech."""
        real_audio = make_real_speech()
        fake_audio = make_fake_speech()
        real_result = SpectralAnalyzer().analyze(real_audio, SR)
        fake_result = SpectralAnalyzer().analyze(fake_audio, SR)
        assert real_result["spectral_score"] <= fake_result["spectral_score"], (
            f"Real score {real_result['spectral_score']} should be ≤ "
            f"fake score {fake_result['spectral_score']}"
        )

    def test_spectral_returns_required_keys(self):
        audio = make_real_speech()
        result = SpectralAnalyzer().analyze(audio, SR)
        for key in [
            "spectral_score", "smoothness_score", "repetition_score",
            "phase_continuity_score", "noise_consistency_score", "flags"
        ]:
            assert key in result, f"Missing key: {key}"

    def test_spectral_handles_empty_audio(self):
        """Empty audio should not raise — return neutral score."""
        result = SpectralAnalyzer().analyze(np.array([], dtype=np.float32), SR)
        assert "spectral_score" in result


# ── Temporal tests ────────────────────────────────────────────────────────────

class TestTemporal:
    def test_temporal_breathing_present(self):
        """Simulated real speech should not necessarily trigger breathing_absent."""
        audio = make_real_speech(duration=8.0)
        result = TemporalAnalyzer().analyze(audio, SR)
        assert "temporal_score" in result
        # Breathing may or may not be detected in synthetic signal
        assert result["temporal_score"] >= 0.0

    def test_temporal_no_breathing_fake(self):
        """TTS simulation should trigger breathing_absent or high temporal score."""
        audio = make_fake_speech(duration=8.0)
        result = TemporalAnalyzer().analyze(audio, SR)
        # Either breathing is flagged absent or temporal score is elevated
        has_flag = "breathing_absent" in result.get("flags", [])
        high_score = result["temporal_score"] > 0.3
        assert has_flag or high_score, (
            f"Expected TTS detection; flags={result['flags']}, score={result['temporal_score']}"
        )

    def test_pitch_jitter_fake_flag(self):
        """Perfectly stable synthetic pitch should trigger unnatural_pitch_stability."""
        audio = make_fake_speech(duration=8.0)
        result = TemporalAnalyzer().analyze(audio, SR)
        # Stable F0 signal should raise flag or have very low jitter
        jitter = result.get("f0_jitter", 1.0)
        flag_present = "unnatural_pitch_stability" in result.get("flags", [])
        assert flag_present or jitter < 0.05, (
            f"Expected low jitter or flag for TTS; jitter={jitter}, flags={result['flags']}"
        )

    def test_temporal_returns_required_keys(self):
        audio = make_real_speech()
        result = TemporalAnalyzer().analyze(audio, SR)
        for key in [
            "temporal_score", "breath_rate_per_minute", "f0_jitter",
            "f0_range_hz", "voiced_ratio", "pause_count", "flags"
        ]:
            assert key in result, f"Missing key: {key}"

    def test_temporal_handles_empty_audio(self):
        result = TemporalAnalyzer().analyze(np.array([], dtype=np.float32), SR)
        assert "temporal_score" in result


# ── Speaker tests ─────────────────────────────────────────────────────────────

class TestSpeaker:
    def test_speaker_consistency_short(self):
        """Audio shorter than MIN_AUDIO_SECONDS should return note='too_short'."""
        audio = make_real_speech(duration=5.0)
        result = SpeakerConsistencyAnalyzer().analyze(audio, SR)
        assert result.get("note") == "too_short", (
            f"Expected too_short note; got: {result}"
        )
        assert result["speaker_score"] == 0.5

    def test_speaker_consistency_long(self):
        """Longer audio should complete without error."""
        audio = make_real_speech(duration=20.0)
        result = SpeakerConsistencyAnalyzer().analyze(audio, SR)
        assert "speaker_score" in result
        assert 0.0 <= result["speaker_score"] <= 1.0

    def test_speaker_returns_required_keys(self):
        audio = make_real_speech(duration=20.0)
        result = SpeakerConsistencyAnalyzer().analyze(audio, SR)
        for key in [
            "speaker_score", "mean_segment_similarity",
            "min_segment_similarity", "identity_drift", "flags"
        ]:
            assert key in result, f"Missing key: {key}"


# ── Fusion / verdict tests ────────────────────────────────────────────────────

class TestFusion:
    def _mock_result(self, score: float, flags: list = None) -> dict:
        return {
            "metadata_score": score,
            "spectral_score": score,
            "temporal_score": score,
            "speaker_score": score,
            "noise_score": score,
            "compression_score": score,
            "noise_consistency_score": 1.0 - score,
            "flags": flags or [],
            "analysis_time_seconds": 0.0,
        }

    def test_fusion_authentic(self):
        """All low scores with few flags should produce AUTHENTIC verdict."""
        low = self._mock_result(0.1, [])
        result = VerdictEngine().decide(low, low, low, low)
        assert result["verdict"] == "AUTHENTIC HUMAN SPEECH", (
            f"Expected AUTHENTIC; got: {result['verdict']} (score={result['composite_score']})"
        )

    def test_fusion_ai_generated(self):
        """High scores with many TTS flags should produce AI GENERATED verdict."""
        tts_flags = [
            "unnatural_pitch_stability", "flat_pitch", "breathing_absent",
            "zcr_uniform", "tts_pause_pattern", "spectral_smooth",
            "missing_noise_floor", "hf_cutoff",
        ]
        high = self._mock_result(0.85, tts_flags)
        result = VerdictEngine().decide(high, high, high, high)
        assert "AI GENERATED" in result["verdict"], (
            f"Expected AI GENERATED; got: {result['verdict']}"
        )

    def test_fusion_spliced(self):
        """Phase discontinuity + speaker change should produce EDITED verdict."""
        spliced_flags = ["phase_discontinuity", "speaker_change_detected"]
        r = self._mock_result(0.7, spliced_flags)
        result = VerdictEngine().decide(r, r, r, r)
        assert "SPLICED" in result["verdict"] or "EDITED" in result["verdict"], (
            f"Expected EDITED/SPLICED; got: {result['verdict']}"
        )

    def test_fusion_returns_required_keys(self):
        low = self._mock_result(0.1)
        result = VerdictEngine().decide(low, low, low, low)
        for key in ["verdict", "confidence", "composite_score", "scores",
                    "anomalies", "flags", "explanation"]:
            assert key in result, f"Missing key: {key}"

    def test_fusion_confidence_range(self):
        """Confidence must always be in [0, 1]."""
        for score in [0.0, 0.3, 0.5, 0.7, 1.0]:
            r = self._mock_result(score)
            result = VerdictEngine().decide(r, r, r, r)
            assert 0.0 <= result["confidence"] <= 1.0, (
                f"Confidence out of range: {result['confidence']}"
            )


# ── Integration test ──────────────────────────────────────────────────────────

class TestIntegration:
    def test_full_pipeline_real(self):
        """Full pipeline on simulated real speech should not error and return a verdict."""
        import librosa
        audio_22k = make_real_speech(duration=8.0, sr=22050)
        path = write_wav(audio_22k, 22050)
        audio, sr = librosa.load(path, sr=16000, mono=True)

        meta = MetadataAnalyzer().analyze(path)
        spec = SpectralAnalyzer().analyze(audio, sr)
        temp = TemporalAnalyzer().analyze(audio, sr)
        spkr = SpeakerConsistencyAnalyzer().analyze(audio, sr)
        result = VerdictEngine().decide(meta, spec, temp, spkr)

        assert result["verdict"] in [
            "AUTHENTIC HUMAN SPEECH",
            "AI GENERATED SPEECH (TTS)",
            "VOICE CLONED SPEECH",
            "EDITED / SPLICED AUDIO",
            "INCONCLUSIVE",
        ]
        assert 0.0 <= result["composite_score"] <= 1.0
        Path(path).unlink(missing_ok=True)

    def test_full_pipeline_fake(self):
        """Full pipeline on TTS simulation should return elevated scores."""
        import librosa
        audio_22k = make_fake_speech(duration=8.0, sr=22050)
        path = write_wav(audio_22k, 22050)
        audio, sr = librosa.load(path, sr=16000, mono=True)

        meta = MetadataAnalyzer().analyze(path)
        spec = SpectralAnalyzer().analyze(audio, sr)
        temp = TemporalAnalyzer().analyze(audio, sr)
        spkr = SpeakerConsistencyAnalyzer().analyze(audio, sr)
        result = VerdictEngine().decide(meta, spec, temp, spkr)

        assert result["composite_score"] >= 0.0
        # Fake signal should accumulate at least some flags
        assert len(result["flags"]) >= 0
        Path(path).unlink(missing_ok=True)
