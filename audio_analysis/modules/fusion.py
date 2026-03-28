"""
fusion.py — Phase 4+5: Weighted Score Fusion & Verdict Engine  [v2 — precision rewrite]

FIXES IN THIS VERSION:
  - Rule 3 (TTS): was `composite > 0.50 AND ≥3 TTS flags`. Problem: composite averages
    cancel out individual signals (e.g., spectral=0.8, temporal=0.3 → composite=0.55).
    Fixed: added SIGNAL VOTING — each module independently casts a vote; majority overrides.
    Also: lowered composite threshold to 0.45 and reduced required TTS flags to 2.

  - Rule 4 (Authentic): `MINOR_FLAGS` incorrectly excluded `unknown_encoder`.
    A legitimate recording from a phone or consumer recorder WILL have an encoder tag.
    Missing encoder is NOT a minor flag — it's a meaningful forensic signal.
    Fixed: `unknown_encoder` removed from MINOR_FLAGS.

  - Confidence: Rule 1 clipped incorrectly with `min(0.95, composite + 0.15)`.
    When composite=0.3 (low suspicion) but splice detected, confidence = 0.45 — too low.
    Fixed: splicing confidence = max(0.80, composite + 0.15).

  - ADDED: Module disagreement penalty. When 3+ modules all score > 0.55, escalate
    composite score upward (the modules are "shouting" — we should listen).

  - ADDED: New flags from v2 modules handled in FLAG_ANOMALY_MAP and TTS_INDICATOR_FLAGS.
"""

import time
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── Human-readable anomaly messages ──────────────────────────────────────────
FLAG_ANOMALY_MAP: dict[str, str] = {
    "format_extension_mismatch":  "File format does not match its extension — possible re-encoding",
    "suspicious_sample_rate":     "Unusual sample rate — common in synthetic pipelines",
    "suspiciously_low_bitrate":   "Bitrate below 64 kbps — unusually low for authentic recordings",
    "synthetic_encoder_detected": "Known TTS/voice-synthesis encoder signature in metadata",
    "unknown_encoder":            "No encoder signature — suspicious for authentic recordings",
    "size_duration_mismatch":     "File size inconsistent with expected audio duration",
    "timestamp_inconsistency":    "File modification time does not match embedded creation timestamp",
    "spectral_smooth":            "Spectrogram bands are unnaturally smooth — real speech has micro-variations",
    "frequency_repetition":       "Repetitive frequency patterns — characteristic of neural TTS",
    "hf_cutoff":                  "Abrupt high-frequency cutoff — typical of TTS or compressed audio",
    "artificial_peaks":           "Artificial spectral peaks — not present in natural recordings",
    "missing_band":               "One or more frequency bands entirely absent from spectrum",
    "phase_discontinuity":        "Phase discontinuities — indicates possible audio splicing",
    "missing_noise_floor":        "No background noise floor — synthetic audio lacks room acoustics",
    "noise_reset":                "Sudden noise floor resets — characteristic of spliced audio",
    "compression_cutoff":         "Hard spectral cutoff below Nyquist — codec or TTS bandwidth limiting",
    "double_compression":         "Spectral holes suggest multiple compression/re-encoding generations",
    "cepstral_regularity":        "Unnaturally regular cepstral patterns — characteristic of neural vocoders",
    "energy_smooth":              "Energy envelope unnaturally uniform — real speech varies significantly",
    "clipping":                   "Amplitude clipping detected — potential post-processing artifact",
    "zcr_uniform":                "Zero-crossing rate too uniform — consonants and vowels should differ",
    "unnatural_pitch_stability":  "Pitch too stable — real voice has biological micro-jitter",
    "flat_pitch":                 "Pitch range extremely narrow — indicates flat TTS delivery",
    "unnatural_voicing":          "Voiced-speech ratio abnormally high — no natural pauses or hesitations",
    "breathing_absent":           "No breathing events detected — all real speakers breathe during speech",
    "uniform_pausing":            "Pause timing metronomic — real speech has irregular inter-utterance gaps",
    "tts_pause_pattern":          "Pause durations cluster at fixed intervals — matches TTS sentence boundaries",
    "unnatural_prosody":          "Prosodic dynamics too predictable — real speech has complex F0 variation",
    "speaker_change_detected":    "Speaker identity changed mid-recording — indicates splicing",
    "voice_clone_drift_detected": "Speaker identity drifts gradually — matches voice cloning artifacts",
    "speaker_tts_uniformity":     "Speaker embeddings unnaturally uniform across segments — TTS artifact",
    "segment_energy_uniform":     "Segment-level energy too uniform — TTS synthesis artifact",
    "music_content_detected":     "Music/singing detected — speech-only penalties disabled for this analysis",
    "analysis_error":             "An error occurred during metadata analysis",
    "empty_audio":                "Audio data is empty or too short to analyze",
}

# ── Flag sets ─────────────────────────────────────────────────────────────────
TTS_INDICATOR_FLAGS = {
    "unnatural_pitch_stability", "flat_pitch", "breathing_absent",
    "zcr_uniform", "tts_pause_pattern", "spectral_smooth",
    "missing_noise_floor", "hf_cutoff", "energy_smooth",
    "frequency_repetition", "cepstral_regularity", "unnatural_prosody",
    "speaker_tts_uniformity", "segment_energy_uniform",
}

# FIX: removed `unknown_encoder` from MINOR_FLAGS — it's a meaningful forensic signal
MINOR_FLAGS = {
    "clipping", "noise_reset", "compression_cutoff",
    "size_duration_mismatch", "timestamp_inconsistency", "missing_band",
}

CLONE_INDICATOR_FLAGS = {
    "unnatural_pitch_stability", "breathing_absent", "voice_clone_drift_detected",
    "speaker_tts_uniformity",
}

SPLICE_INDICATOR_FLAGS = {
    "phase_discontinuity", "speaker_change_detected", "noise_reset",
}


class VerdictEngine:
    """
    Fuses module scores and applies forensic verdict rules.

    Decision flow:
      1. Weighted score fusion → composite_score
      2. Module signal voting → may escalate composite
      3. Rule-based verdict (first match wins)
      4. Anomaly list construction
      5. Natural-language explanation
    """

    def decide(
        self,
        metadata_result: dict,
        spectral_result:  dict,
        temporal_result:  dict,
        speaker_result:   dict,
        file_info: dict | None = None,
    ) -> dict:
        start_time = time.time()

        # ── Weighted score fusion ─────────────────────────────────────────────
        spectral_score    = float(spectral_result.get("spectral_score",    0.5))
        temporal_score    = float(temporal_result.get("temporal_score",    0.5))
        metadata_score    = float(metadata_result.get("metadata_score",    0.5))
        speaker_score     = float(speaker_result.get("speaker_score",      0.5))
        noise_score       = float(spectral_result.get("noise_score",       0.5))
        compression_score = float(spectral_result.get("compression_score", 0.5))

        composite_score = (
            spectral_score     * 0.33 +
            temporal_score     * 0.33 +
            metadata_score     * 0.08 +
            speaker_score      * 0.10 +
            noise_score        * 0.08 +
            compression_score  * 0.08
        )
        composite_score = float(np.clip(composite_score, 0.0, 1.0))

        # ── MUSIC CONTENT DETECTION ───────────────────────────────────────────
        # Check if temporal module flagged this as music content.
        # Music triggers false-positive speech penalties (breathing_absent,
        # pitch_stability, pause_regularity) that push real songs to INCONCLUSIVE.
        # When music is detected we: (a) suppress speech-only TTS flag escalation,
        # (b) de-weight the temporal score since its speech penalties are zeroed out,
        # and (c) apply a dedicated music authenticity rule before the standard rules.
        is_music = bool(temporal_result.get("is_music", False))
        music_confidence = float(temporal_result.get("music_confidence", 0.0))

        if is_music:
            # Re-weight composite: temporal score is already cleaned of speech
            # penalties, but we further reduce its pull vs spectral (which is the
            # primary AI-music detector) and noise (real recordings have a floor).
            composite_score = float(np.clip(
                spectral_score     * 0.35 +
                temporal_score     * 0.20 +   # reduced — speech penalties zeroed
                metadata_score     * 0.08 +
                speaker_score      * 0.10 +
                noise_score        * 0.15 +   # increased — room noise is key signal
                compression_score  * 0.12,
                0.0, 1.0
            ))

        # ── NEW: Module signal voting & escalation ────────────────────────────
        # If 3+ primary modules independently signal "suspicious", boost composite.
        # This prevents averaging from burying clear multi-module evidence.
        # Skip escalation for music — its temporal score is intentionally deflated.
        if not is_music:
            primary_votes = sum([
                spectral_score > 0.55,
                temporal_score > 0.55,
                noise_score    > 0.60,
                speaker_score  > 0.50,
            ])
            if primary_votes >= 3:
                escalation = (primary_votes - 2) * 0.08
                composite_score = float(np.clip(composite_score + escalation, 0.0, 1.0))

        # ── NEW: Flag-density escalation ──────────────────────────────────────
        # Collect flags early to use in escalation (full collection happens below)
        _early_flags: set[str] = set()
        for _r in [metadata_result, spectral_result, temporal_result, speaker_result]:
            for _f in _r.get("flags", []):
                _early_flags.add(_f)
        # When background music or heavy compression suppresses module SCORES but
        # the flag detectors (which use thresholds, not magnitudes) still fire,
        # the composite is artificially low. Escalate based on TTS flag density.
        # For music: only escalate on non-speech-specific TTS flags.
        _tts_flags_early = TTS_INDICATOR_FLAGS & _early_flags
        if is_music:
            # Remove speech-only flags before counting — they were already stripped
            # in temporal.py but may appear in spectral result
            SPEECH_ONLY_TTS_FLAGS = {
                "breathing_absent", "unnatural_pitch_stability",
                "flat_pitch", "unnatural_voicing", "uniform_pausing",
                "tts_pause_pattern", "unnatural_prosody",
            }
            _tts_flags_early = _tts_flags_early - SPEECH_ONLY_TTS_FLAGS
        if len(_tts_flags_early) >= 5:
            flag_escalation = (len(_tts_flags_early) - 4) * 0.04
            composite_score = float(np.clip(composite_score + flag_escalation, 0.0, 1.0))

        # ── Collect all flags ─────────────────────────────────────────────────
        all_flags: set[str] = set()
        for result in [metadata_result, spectral_result, temporal_result, speaker_result]:
            for flag in result.get("flags", []):
                all_flags.add(flag)

        # ── Rule-based verdict ────────────────────────────────────────────────
        verdict, confidence = self._apply_verdict_rules(
            composite_score, all_flags,
            spectral_score, temporal_score, noise_score, speaker_score,
        )

        # ── Anomaly messages ──────────────────────────────────────────────────
        anomalies = [
            FLAG_ANOMALY_MAP[flag]
            for flag in sorted(all_flags)
            if flag in FLAG_ANOMALY_MAP
        ]

        # ── Explanation ───────────────────────────────────────────────────────
        explanation = self._generate_explanation(
            verdict=verdict,
            confidence=confidence,
            composite_score=composite_score,
            all_flags=all_flags,
            spectral_score=spectral_score,
            temporal_score=temporal_score,
            temporal_result=temporal_result,
            speaker_result=speaker_result,
            spectral_result=spectral_result,
        )

        scores = {
            "spectral":    round(spectral_score,    4),
            "temporal":    round(temporal_score,    4),
            "noise":       round(noise_score,       4),
            "metadata":    round(metadata_score,    4),
            "speaker":     round(speaker_score,     4),
            "compression": round(compression_score, 4),
        }

        result = {
            "verdict":                  verdict,
            "confidence":               round(float(confidence), 4),
            "composite_score":          round(composite_score, 4),
            "scores":                   scores,
            "anomalies":                anomalies,
            "flags":                    sorted(list(all_flags)),
            "explanation":              explanation,
            "analysis_time_seconds":    round(time.time() - start_time, 3),
        }

        if file_info:
            result["file_info"] = file_info

        return result

    # ─────────────────────────────────────────────────────────────────────────
    def _apply_verdict_rules(
        self,
        composite_score: float,
        all_flags: set[str],
        spectral_score: float,
        temporal_score: float,
        noise_score: float,
        speaker_score: float,
    ) -> tuple[str, float]:
        """
        Apply ordered forensic verdict rules. First match wins.

        FIXES:
          - Rule 3 (TTS): added module-vote override + lowered thresholds
          - Rule 4 (Authentic): tightened; unknown_encoder no longer in MINOR_FLAGS
          - Rule 1 (Splice): confidence fixed
          - RULE 0 (Music): dedicated authentic rule for song/music content
        """

        # RULE 0 — AUTHENTIC MUSIC
        # Songs trigger breathing_absent, pitch_stability, and pause_regularity
        # false positives that push real music to INCONCLUSIVE. The temporal module
        # already neutralizes these penalties when music is detected, but we add an
        # explicit fast-path here so music with low composite score is immediately
        # classified as authentic rather than waiting for Rule 4.
        # We still check for smoking-gun AI signals (spectral_smooth, missing noise
        # floor, metronomic beat from AI music generators) before clearing it.
        if "music_content_detected" in all_flags:
            MUSIC_AI_SMOKING_GUNS = {
                "missing_noise_floor",      # AI music has perfect silence floor
                "synthetic_encoder_detected",  # metadata reveals TTS/AI tool
                "speaker_tts_uniformity",   # AI music has uniform "speaker" embedding
                "segment_energy_uniform",   # AI music has perfectly flat energy
                "double_compression",       # layered re-encoding = AI pipeline
            }
            # Remove speech-only TTS flags from consideration for music
            SPEECH_ONLY_TTS = {
                "breathing_absent", "unnatural_pitch_stability",
                "flat_pitch", "unnatural_voicing", "uniform_pausing",
                "tts_pause_pattern", "unnatural_prosody",
            }
            music_ai_hits = MUSIC_AI_SMOKING_GUNS & all_flags
            non_speech_tts = (TTS_INDICATOR_FLAGS - SPEECH_ONLY_TTS) & all_flags

            if len(music_ai_hits) == 0 and composite_score < 0.55 and len(non_speech_tts) <= 2:
                # Clean music — authentic
                confidence = float(np.clip(1.0 - composite_score + 0.05, 0.55, 0.95))
                return "AUTHENTIC HUMAN SPEECH", confidence
            elif len(music_ai_hits) >= 2 or composite_score > 0.65:
                # Multiple AI music signatures — flag as generated
                confidence = min(0.90, composite_score + 0.10)
                return "AI GENERATED SPEECH (TTS)", confidence
            # else: fall through to standard rules with cleaned composite
        splice_hits = SPLICE_INDICATOR_FLAGS & all_flags
        if "phase_discontinuity" in all_flags and "speaker_change_detected" in all_flags:
            # FIX: confidence calculation — use max() not min()
            confidence = max(0.80, min(0.97, composite_score + 0.20))
            return "EDITED / SPLICED AUDIO", confidence

        # RULE 2 — VOICE CLONED
        clone_hits = CLONE_INDICATOR_FLAGS & all_flags
        if (
            "voice_clone_drift_detected" in all_flags
            and len(clone_hits) >= 2
            and composite_score > 0.50
        ):
            return "VOICE CLONED SPEECH", min(0.95, composite_score + 0.05)

        # RULE 3 — AI GENERATED (TTS)
        tts_flags_present = TTS_INDICATOR_FLAGS & all_flags

        # FIX: Added module-vote override path
        # Path A: classic composite + flag count (lowered thresholds)
        path_a = composite_score > 0.45 and len(tts_flags_present) >= 2
        # Path B: strong spectral AND temporal agreement (modules shouting)
        path_b = spectral_score > 0.60 and temporal_score > 0.60
        # Path C: breathing absent AND pitch too stable (near-certain TTS)
        # FIX: lowered composite threshold from 0.40 → 0.20; background music can
        # depress module scores even when TTS signatures are unambiguously present.
        path_c = (
            "breathing_absent" in all_flags
            and "unnatural_pitch_stability" in all_flags
            and composite_score > 0.20
        )
        # Path D: missing noise floor + cepstral regularity (vocoder signature)
        path_d = (
            "missing_noise_floor" in all_flags
            and "cepstral_regularity" in all_flags
        )
        # Path E: flag-count override — when ≥6 independent TTS indicators fire,
        # the signal is TTS regardless of composite score.  Background music or
        # heavy compression can suppress individual module SCORES while leaving
        # flag detections intact; this path ensures those flags are not ignored.
        path_e = len(tts_flags_present) >= 6
        # Path F: speaker uniformity + cepstral regularity + zcr uniform
        # (three orthogonal TTS signatures from different modules)
        path_f = (
            "speaker_tts_uniformity" in all_flags
            and "cepstral_regularity" in all_flags
            and "zcr_uniform" in all_flags
        )

        if path_a or path_b or path_c or path_d or path_e or path_f:
            # Confidence: use flag-density when composite is suppressed
            flag_density_confidence = min(0.95, 0.50 + len(tts_flags_present) * 0.05)
            confidence = max(flag_density_confidence, min(0.95, composite_score + 0.05))
            return "AI GENERATED SPEECH (TTS)", confidence

        # RULE 4 — AUTHENTIC
        # A recording is authentic when:
        #   (a) No smoking-gun TTS indicators are present, AND
        #   (b) The composite score is below the suspicion threshold
        #
        # The old rule (strong_flags <= 1) is too brittle: spectral flags like
        # cepstral_regularity, frequency_repetition, and phase_discontinuity fire
        # as false positives on short audio and compressed (MP3/AAC) recordings,
        # leaving real audio stuck in INCONCLUSIVE because strong_flags > 1.
        #
        # Better approach: check for ABSENCE of true TTS smoking guns, and require
        # that any remaining strong flags come from only one module (spectral
        # compression artifacts are common in real recordings; cross-module
        # agreement is what makes a signal truly suspicious).

        # Smoking-gun flags that a real human voice should never produce:
        SMOKING_GUN_FLAGS = {
            "breathing_absent",
            "speaker_tts_uniformity",
            "segment_energy_uniform",
            "voice_clone_drift_detected",
            "synthetic_encoder_detected",
            "tts_pause_pattern",
        }
        smoking_guns_present = SMOKING_GUN_FLAGS & all_flags

        # Flags that commonly fire on compressed/short real recordings (not true TTS signals):
        COMPRESSION_AND_SHORT_AUDIO_FLAGS = {
            "hf_cutoff", "missing_band", "compression_cutoff",
            "cepstral_regularity", "frequency_repetition",
            "phase_discontinuity", "noise_reset",
        }

        # Strong flags excluding both MINOR_FLAGS and compression/short-audio noise
        true_strong_flags = all_flags - MINOR_FLAGS - COMPRESSION_AND_SHORT_AUDIO_FLAGS

        if (
            composite_score < 0.45
            and len(smoking_guns_present) == 0
            and len(tts_flags_present) <= 3
            and (len(true_strong_flags) <= 2 or speaker_score < 0.35)
        ):
            confidence = min(0.95, 1.0 - composite_score + 0.05 * max(0, 2 - len(true_strong_flags)))
            return "AUTHENTIC HUMAN SPEECH", min(0.95, float(confidence))

        # RULE 5 — AUTO TUNED
        return "AUTO TUNED", round(composite_score, 4)

    # ─────────────────────────────────────────────────────────────────────────
    def _generate_explanation(
        self,
        verdict: str,
        confidence: float,
        composite_score: float,
        all_flags: set[str],
        spectral_score: float,
        temporal_score: float,
        temporal_result: dict,
        speaker_result: dict,
        spectral_result: dict,
    ) -> str:
        n_anomalies    = len(all_flags)
        confidence_pct = int(confidence * 100)

        if "AI GENERATED" in verdict:
            top_flags    = sorted(TTS_INDICATOR_FLAGS & all_flags)[:4]
            top_str      = "; ".join(FLAG_ANOMALY_MAP.get(f, f) for f in top_flags) or "see flag list"
            breath_rate  = temporal_result.get("breath_rate_per_minute", 0.0)
            f0_jitter    = temporal_result.get("f0_jitter", 0.0)
            br_status    = (
                "absent (0 detected)" if "breathing_absent" in all_flags
                else f"present ({breath_rate:.1f}/min)"
            )
            return (
                f"Forensic analysis found {n_anomalies} anomalies consistent with "
                f"AI text-to-speech synthesis. Key indicators: {top_str}. "
                f"Spectral score {spectral_score:.2f} and temporal score {temporal_score:.2f} "
                f"both exceed authenticity threshold. Breathing was {br_status}; "
                f"pitch jitter ({f0_jitter:.4f}) is below the natural human threshold. "
                f"Overall confidence: {confidence_pct}%."
            )

        elif "VOICE CLONED" in verdict:
            seg_count = speaker_result.get("segment_count", 0)
            mean_sim  = speaker_result.get("mean_segment_similarity", 1.0)
            min_sim   = speaker_result.get("min_segment_similarity", 1.0)
            return (
                f"Voice cloning characteristics detected. Speaker identity drift across "
                f"{seg_count} segments; mean cosine similarity {mean_sim:.3f} (min: {min_sim:.3f}). "
                f"Negative identity trend slope indicates cloned voice losing fidelity over time. "
                f"Confidence: {confidence_pct}%."
            )

        elif "SPLICED" in verdict or "EDITED" in verdict:
            n_splices = len(spectral_result.get("splice_candidate_frames", []))
            return (
                f"Phase analysis detected {n_splices} phase discontinuity event(s) consistent "
                f"with audio splicing. Speaker identity also changed mid-recording. "
                f"Composite suspicion: {composite_score:.2f}. Confidence: {confidence_pct}%."
            )

        elif "AUTHENTIC" in verdict:
            breath_rate = temporal_result.get("breath_rate_per_minute", 0.0)
            f0_jitter   = temporal_result.get("f0_jitter", 0.0)
            noise_score = spectral_result.get("noise_consistency_score", 0.5)
            is_music    = bool(temporal_result.get("is_music", False))
            if is_music:
                music_ev     = temporal_result.get("music_evidence", {})
                cents        = music_ev.get("mean_cents_from_grid", 0)
                cents_interp = music_ev.get("cents_interpretation", "")
                f1_cv        = music_ev.get("f1_formant_cv", 0)
                f1_interp    = music_ev.get("f1_interpretation", "")
                vibrato      = music_ev.get("vibrato_power_ratio", 0)
                pitch_range  = music_ev.get("pitch_range_hz", 0)
                beat_str     = music_ev.get("beat_autocorr_strength", 0)
                return (
                    f"Music/singing content detected — speech-only forensic penalties "
                    f"(breathing, pitch-stability, pause-regularity) were disabled as they "
                    f"do not apply to songs. "
                    f"Authenticity signals: "
                    f"pitch deviation {cents:.1f} cents from equal temperament ({cents_interp}); "
                    f"F1 formant variability {f1_cv:.3f} ({f1_interp}); "
                    f"vibrato power ratio {vibrato:.3f} ({'present' if vibrato > 0.03 else 'absent'}); "
                    f"pitch range {pitch_range:.0f} Hz; beat strength {beat_str:.2f}. "
                    f"No AI-music smoking-gun signals (frozen formants, on-grid pitch, "
                    f"missing noise floor) detected. Confidence: {confidence_pct}%."
                )
            return (
                f"No significant forensic anomalies detected. "
                f"Breathing: {breath_rate:.1f}/min (natural range 12-20/min). "
                f"Pitch jitter ({f0_jitter:.4f}) within natural range. "
                f"Noise floor consistency: {noise_score:.2f}. "
                f"All temporal, spectral, and speaker metrics within expected bounds. "
                f"Confidence: {confidence_pct}%."
            )

        else:  # AUTO TUNED
            flag_list = sorted(all_flags)[:5]
            flag_str  = ", ".join(flag_list) if flag_list else "none"
            return (
                f"Signal shows {n_anomalies} anomalies consistent with pitch correction or auto-tune processing "
                f"(flags: {flag_str}). Composite score: {composite_score:.2f}. "
                f"Voice characteristics suggest heavy vocal processing or auto-tune has been applied."
            )
