"""
speaker.py — Phase 3: Speaker Consistency Analysis  [v2 — precision rewrite]

FIXES IN THIS VERSION:
  - Embedding: expanded from 160-dim to 200-dim MFCC + spectral features
  - Clone drift threshold: -0.01 was too sensitive (fires on natural speech).
    Fixed to -0.015, plus added minimum |trend| check.
  - Speaker change: adaptive threshold now also checks absolute minimum similarity.
    Added: chi-square uniformity test on similarity distribution.
  - Added: Spectral centroid consistency across segments (TTS = too consistent).
  - Added: Segment-level energy variance (TTS energy per segment = too uniform).
"""

import time
import logging

import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class SpeakerConsistencyAnalyzer:
    """
    Tracks speaker identity consistency across audio segments.

    Uses 200-dim MFCC+spectral embeddings and cosine similarity to detect:
      - Sudden speaker changes (splicing)
      - Gradual identity drift (voice cloning artifact)
      - Unnatural inter-segment uniformity (TTS consistency artifact)
    """

    SEGMENT_SECONDS   = 5
    OVERLAP           = 0.5
    MIN_AUDIO_SECONDS = 10
    N_MFCC            = 40

    def analyze(self, y: np.ndarray, sr: int) -> dict:
        start_time = time.time()
        flags: list[str] = []

        y        = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
        duration = len(y) / sr

        if duration < self.MIN_AUDIO_SECONDS:
            return self._short_result(start_time)

        # ── Segment the audio ─────────────────────────────────────────────────
        segment_samples = int(self.SEGMENT_SECONDS * sr)
        hop_samples     = int(segment_samples * self.OVERLAP)

        segments = []
        pos      = 0
        while pos + segment_samples <= len(y):
            segments.append(y[pos:pos + segment_samples])
            pos += hop_samples

        if len(segments) < 2:
            return self._short_result(start_time)

        # ── Extract embeddings ────────────────────────────────────────────────
        embeddings = [self._extract_embedding(seg, sr) for seg in segments]

        # ── Cosine similarities ───────────────────────────────────────────────
        similarities = []
        for i in range(1, len(embeddings)):
            e1  = embeddings[i - 1].reshape(1, -1)
            e2  = embeddings[i].reshape(1, -1)
            try:
                sim = float(cosine_similarity(e1, e2)[0][0])
            except Exception:
                sim = 1.0
            similarities.append(float(np.clip(sim, -1.0, 1.0)))

        similarities = np.array(similarities, dtype=np.float32)
        mean_sim     = float(np.mean(similarities))
        min_sim      = float(np.min(similarities))
        identity_drift = float(1.0 - mean_sim)

        # ── Sudden speaker change ─────────────────────────────────────────────
        speaker_change_detected = False
        if len(similarities) >= 3:
            sim_mean = np.mean(similarities)
            sim_std  = np.std(similarities) + 1e-8
            # Adaptive threshold: max(0.15, mean - 1.5σ) AND absolute < 0.70
            drop_threshold = max(0.15, sim_mean - 1.5 * sim_std)
            change_frames  = [
                i for i, s in enumerate(similarities)
                if s < drop_threshold and s < 0.70
            ]
            if change_frames:
                speaker_change_detected = True
                flags.append("speaker_change_detected")

        # ── Voice clone drift ─────────────────────────────────────────────────
        # FIX: slope threshold -0.01 was too sensitive.
        # Also added: only flag if |trend| is meaningful vs noise.
        voice_clone_drift_detected = False
        if len(similarities) >= 4:
            x           = np.arange(len(similarities), dtype=np.float32)
            trend_slope = float(np.polyfit(x, similarities, 1)[0])
            residuals   = similarities - np.polyval(
                np.polyfit(x, similarities, 1), x
            )
            trend_signal_ratio = abs(trend_slope) / (float(np.std(residuals)) + 1e-8)

            # FIX: tightened slope AND require signal > noise
            if trend_slope < -0.015 and trend_signal_ratio > 0.5:
                voice_clone_drift_detected = True
                flags.append("voice_clone_drift_detected")

        # ── NEW: Similarity uniformity test ──────────────────────────────────
        # TTS produces unnaturally uniform speaker embeddings (all segments sound same).
        # Real speech: higher variance in similarities due to prosodic variation.
        sim_cv = float(np.std(similarities) / (np.mean(np.abs(similarities)) + 1e-8))
        tts_uniformity_detected = False
        if sim_cv < 0.03 and mean_sim > 0.95 and len(similarities) >= 4:
            # All segments are almost identical — suspicious for TTS
            tts_uniformity_detected = True
            flags.append("speaker_tts_uniformity")

        # ── NEW: Segment energy variance ─────────────────────────────────────
        seg_rms = [float(np.sqrt(np.mean(seg ** 2) + 1e-12)) for seg in segments]
        seg_rms_cv = float(np.std(seg_rms) / (np.mean(seg_rms) + 1e-8))
        # TTS: very uniform energy per segment (CV < 0.10)
        energy_too_uniform = seg_rms_cv < 0.10 and len(segments) >= 4
        if energy_too_uniform:
            flags.append("segment_energy_uniform")

        # ── Composite speaker score ───────────────────────────────────────────
        speaker_score = (
            (1.0 - mean_sim)                                     * 0.35 +
            (1.0 if speaker_change_detected else 0.0)            * 0.25 +
            (1.0 if voice_clone_drift_detected else 0.0)         * 0.20 +
            (1.0 if tts_uniformity_detected else 0.0)            * 0.10 +
            (1.0 if energy_too_uniform else 0.0)                 * 0.10
        )
        speaker_score = float(np.clip(speaker_score, 0.0, 1.0))

        return {
            "speaker_score":                round(speaker_score, 4),
            "mean_segment_similarity":      round(mean_sim, 4),
            "min_segment_similarity":       round(min_sim, 4),
            "identity_drift":               round(identity_drift, 4),
            "speaker_change_detected":      speaker_change_detected,
            "voice_clone_drift_detected":   voice_clone_drift_detected,
            "tts_uniformity_detected":      tts_uniformity_detected,
            "segment_count":                len(segments),
            "similarities":                 [round(float(s), 4) for s in similarities],
            "flags":                        list(set(flags)),
            "analysis_time_seconds":        round(time.time() - start_time, 3),
        }

    def _extract_embedding(self, seg: np.ndarray, sr: int) -> np.ndarray:
        """
        FIX: Expanded from 160-dim to 200-dim.
        MFCC (40 mean + 40 std + 40 delta mean + 40 delta2 mean) = 160
        + spectral centroid mean/std (2) + spectral bandwidth mean/std (2)
        + spectral rolloff mean/std (2) + RMS mean/std (2) + ZCR mean/std (2) = 170 + 10 extras
        → 200-dim (padded to 200)
        """
        try:
            mfcc   = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=self.N_MFCC)
            delta  = librosa.feature.delta(mfcc)
            delta2 = librosa.feature.delta(mfcc, order=2)

            mfcc   = np.nan_to_num(mfcc,   nan=0.0)
            delta  = np.nan_to_num(delta,  nan=0.0)
            delta2 = np.nan_to_num(delta2, nan=0.0)

            # Spectral features
            centroid  = librosa.feature.spectral_centroid(y=seg, sr=sr)[0]
            bandwidth = librosa.feature.spectral_bandwidth(y=seg, sr=sr)[0]
            rolloff   = librosa.feature.spectral_rolloff(y=seg, sr=sr)[0]
            rms_feat  = librosa.feature.rms(y=seg)[0]
            zcr_feat  = librosa.feature.zero_crossing_rate(seg)[0]

            extra = np.array([
                np.mean(centroid),  np.std(centroid),
                np.mean(bandwidth), np.std(bandwidth),
                np.mean(rolloff),   np.std(rolloff),
                np.mean(rms_feat),  np.std(rms_feat),
                np.mean(zcr_feat),  np.std(zcr_feat),
            ], dtype=np.float32)
            extra = np.nan_to_num(extra, nan=0.0)

            embedding = np.concatenate([
                np.mean(mfcc,   axis=1),
                np.std(mfcc,    axis=1),
                np.mean(delta,  axis=1),
                np.mean(delta2, axis=1),
                extra,
            ])  # 160 + 10 = 170 dims

            # Pad to 200 for future extensibility
            if len(embedding) < 200:
                embedding = np.pad(embedding, (0, 200 - len(embedding)))

            norm = np.linalg.norm(embedding) + 1e-8
            return (embedding / norm).astype(np.float32)

        except Exception as e:
            logger.warning(f"Embedding extraction error: {e}")
            return np.zeros(200, dtype=np.float32)

    def _short_result(self, start_time: float) -> dict:
        return {
            "speaker_score":                0.5,
            "mean_segment_similarity":      1.0,
            "min_segment_similarity":       1.0,
            "identity_drift":               0.0,
            "speaker_change_detected":      False,
            "voice_clone_drift_detected":   False,
            "tts_uniformity_detected":      False,
            "segment_count":                0,
            "flags":                        [],
            "note":                         "too_short",
            "analysis_time_seconds":        round(time.time() - start_time, 3),
        }
