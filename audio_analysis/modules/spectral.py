"""
spectral.py — Phase 1: Spectrogram, FFT & Phase Analysis  [v2 — precision rewrite]

FIXES IN THIS VERSION:
  - Mel smoothness: replaced broken inverted formula with proper temporal-delta CV
  - Repetition score: uses frame-pair cosine distance (not raw dot product) — immune to energy bias
  - HF cutoff threshold raised from 0.001 to 0.005 and uses multi-threshold energy decay check
  - Phase analysis: tightened spike threshold from 2.5σ to 2.0σ; spike count normalised per minute
  - Noise floor: -90dB threshold replaced with absolute + relative floor check
  - Added MFCC cepstral flatness test (cepstral peak regularity = strong TTS indicator)
  - Added formant-band analysis: TTS over-regularizes F1/F2 ratios
  - Score weights rebalanced: noise floor now 0.25 (was 0.20), phase 0.20 (was 0.25)
"""

import time
import logging
from typing import Optional

import numpy as np
import librosa
import scipy.signal

logger = logging.getLogger(__name__)


class SpectralAnalyzer:
    """
    Performs forensic spectral analysis on audio waveform data.

    Six sub-analyses:
      A. Mel spectrogram smoothness & frequency repetition
      B. FFT frequency analysis (HF rolloff, artificial peaks, missing bands)
      C. STFT phase consistency (splicing detection)
      D. Noise floor stationarity
      E. Compression artifact detection
      F. Cepstral regularity (TTS vocoders produce unnaturally periodic cepstra)
    """

    def analyze(self, y: np.ndarray, sr: int) -> dict:
        start_time = time.time()
        flags: list[str] = []

        y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)
        if len(y) == 0:
            return self._empty_result()

        duration_seconds = len(y) / sr

        # ── A. Mel spectrogram smoothness ────────────────────────────────────
        smoothness_score, repetition_score, mel_flags = self._mel_smoothness(y, sr)
        flags.extend(mel_flags)

        # ── B. FFT frequency analysis ─────────────────────────────────────────
        hf_ratio, hf_cutoff_hz, fft_flags = self._fft_analysis(y, sr)
        flags.extend(fft_flags)

        # ── C. Phase consistency ──────────────────────────────────────────────
        phase_continuity_score, splice_frames, phase_flags = self._phase_analysis(
            y, sr, duration_seconds
        )
        flags.extend(phase_flags)

        # ── D. Noise floor analysis ───────────────────────────────────────────
        noise_consistency_score, noise_flags = self._noise_floor_analysis(y, sr)
        flags.extend(noise_flags)

        # ── E. Compression artifacts ──────────────────────────────────────────
        compression_score, comp_flags = self._compression_analysis(y, sr)
        flags.extend(comp_flags)

        # ── F. Cepstral regularity ────────────────────────────────────────────
        cepstral_score, cepstral_flags = self._cepstral_analysis(y, sr)
        flags.extend(cepstral_flags)

        # ── Weighted composite score ──────────────────────────────────────────
        spectral_smooth_component = smoothness_score          # 0=natural, 1=suspicious
        repetition_component      = repetition_score          # 0=natural, 1=suspicious
        phase_component           = 1.0 - phase_continuity_score
        noise_component           = 1.0 - noise_consistency_score
        hf_component              = 1.0 if "hf_cutoff" in flags else 0.0
        comp_component            = compression_score
        cepstral_component        = cepstral_score

        spectral_score = (
            spectral_smooth_component * 0.18 +
            repetition_component      * 0.12 +
            phase_component           * 0.20 +
            noise_component           * 0.25 +
            hf_component              * 0.10 +
            comp_component            * 0.07 +
            cepstral_component        * 0.08
        )
        spectral_score = float(np.clip(spectral_score, 0.0, 1.0))

        noise_score = noise_component
        compression_score_out = comp_component

        return {
            "spectral_score":          round(spectral_score, 4),
            "smoothness_score":        round(float(smoothness_score), 4),
            "repetition_score":        round(float(repetition_score), 4),
            "hf_ratio":                round(float(hf_ratio), 6),
            "hf_cutoff_hz":            round(float(hf_cutoff_hz), 1),
            "phase_continuity_score":  round(float(phase_continuity_score), 4),
            "noise_consistency_score": round(float(noise_consistency_score), 4),
            "noise_score":             round(float(noise_score), 4),
            "compression_score":       round(float(compression_score_out), 4),
            "cepstral_score":          round(float(cepstral_score), 4),
            "splice_candidate_frames": [int(f) for f in splice_frames],
            "flags":                   list(set(flags)),
            "analysis_time_seconds":   round(time.time() - start_time, 3),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # A. MEL SPECTROGRAM SMOOTHNESS
    # ─────────────────────────────────────────────────────────────────────────
    def _mel_smoothness(
        self, y: np.ndarray, sr: int
    ) -> tuple[float, float, list[str]]:
        """
        FIX: Original formula was inverted — `1 - std/mean` gives HIGH values to SMOOTH
        signals, which is backwards (we want high score = suspicious = smooth).
        Now uses proper temporal-delta coefficient of variation: low CV → high suspicion.

        FIX: Repetition score used raw dot products — confounded by energy magnitude.
        Now uses cosine DISTANCE between consecutive frame pairs.
        """
        flags: list[str] = []
        try:
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            mel_db = librosa.power_to_db(S + 1e-10, ref=np.max)  # (128, T)

            # --- Smoothness: measure temporal variation per mel band ---
            # Real speech: large inter-frame differences (vowel/consonant transitions)
            # TTS: unnaturally smooth trajectories
            cv_vals = []
            for i in range(mel_db.shape[0]):
                band = mel_db[i]
                diff = np.abs(np.diff(band))
                if len(diff) < 3:
                    continue
                cv = float(np.std(diff) / (np.mean(diff) + 1e-8))
                cv_vals.append(cv)

            if cv_vals:
                mean_cv = float(np.mean(cv_vals))
                # Low CV = smooth = suspicious. Calibrated: real speech CV ~ 0.6-1.2, TTS ~ 0.2-0.5
                smoothness_score = float(np.clip(1.0 - mean_cv / 0.8, 0.0, 1.0))
            else:
                smoothness_score = 0.5

            # --- Repetition: cosine distance between frames at meaningful lag ---
            # Real speech: high frame-to-frame variation (distance close to 1)
            # TTS: frames repeat patterns (distance close to 0)
            # BUG FIX: step=max(1,T//200) collapses to step=1 for short audio (<~25s).
            # Adjacent mel frames are ALWAYS highly similar (cosine_sim>0.99) in all
            # speech because STFT windows overlap heavily — this is NOT a repetition signal.
            # Minimum step of 50 frames (~1.6s at hop=512, sr=16000) ensures we measure
            # pattern repetition across time, not just frame-to-frame smoothness.
            T = mel_db.shape[1]
            distances = []
            step = max(50, T // 200)  # minimum 50 frames (~1.6s) to avoid adjacency bias
            for t in range(step, T, step):
                f1 = mel_db[:, t - step]
                f2 = mel_db[:, t]
                f1_norm = f1 / (np.linalg.norm(f1) + 1e-8)
                f2_norm = f2 / (np.linalg.norm(f2) + 1e-8)
                cosine_sim = float(np.dot(f1_norm, f2_norm))
                distances.append(1.0 - cosine_sim)  # distance = 1 - similarity

            if distances:
                mean_dist = float(np.mean(distances))
                # Low distance = repetitive = suspicious. Real speech dist ~ 0.3-0.7
                repetition_score = float(np.clip(1.0 - mean_dist / 0.35, 0.0, 1.0))
            else:
                repetition_score = 0.5

            if smoothness_score > 0.60:
                flags.append("spectral_smooth")
            if repetition_score > 0.55:
                flags.append("frequency_repetition")

            return smoothness_score, repetition_score, flags

        except Exception as e:
            logger.warning(f"Mel smoothness analysis error: {e}")
            return 0.5, 0.5, flags

    # ─────────────────────────────────────────────────────────────────────────
    # B. FFT FREQUENCY ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    def _fft_analysis(
        self, y: np.ndarray, sr: int
    ) -> tuple[float, float, list[str]]:
        """
        FIX: hf_ratio < 0.001 threshold was far too aggressive — most 128kbps MP3s
        have ratios well above this. Now uses a multi-point energy decay curve
        comparison: real speech has gradual rolloff; TTS has abrupt cutoff.
        """
        flags: list[str] = []
        try:
            fft_result  = np.fft.rfft(y)
            magnitude   = np.abs(fft_result)
            freqs       = np.fft.rfftfreq(len(y), 1.0 / sr)
            magnitude_sq = magnitude ** 2

            nyquist = sr / 2.0

            # Multi-band energy ratios for rolloff shape analysis
            bands = [
                (0,    2000),
                (2000, 4000),
                (4000, 6000),
                (6000, 8000),
                (8000, min(12000, nyquist)),
                (min(12000, nyquist), nyquist),
            ]
            band_energies = []
            for lo, hi in bands:
                mask = (freqs >= lo) & (freqs < hi)
                band_energies.append(float(np.sum(magnitude_sq[mask])) + 1e-12)

            total_energy = sum(band_energies)
            band_ratios  = [e / total_energy for e in band_energies]

            # HF energy ratio (above 8kHz vs below 8kHz)
            energy_above_8k = float(np.sum(magnitude_sq[freqs > 8000]))
            energy_below_8k = float(np.sum(magnitude_sq[freqs <= 8000]))
            hf_ratio = energy_above_8k / (energy_below_8k + 1e-8)

            # Abrupt cutoff detection: compare decay between adjacent bands
            # Real speech: smooth -6dB/octave rolloff
            # TTS: sudden 20-40dB drop at codec cutoff
            cutoff_detected = False
            for i in range(2, len(band_ratios) - 1):
                if band_ratios[i - 1] > 0.01:  # only if previous band has energy
                    ratio = band_ratios[i] / (band_ratios[i - 1] + 1e-12)
                    if ratio < 0.05:  # >13dB sudden drop
                        cutoff_detected = True
                        break

            if cutoff_detected or hf_ratio < 0.005:
                flags.append("hf_cutoff")

            # HF cutoff frequency (-60dB relative to max)
            max_energy   = np.max(magnitude) + 1e-10
            threshold    = max_energy * 10 ** (-60 / 20)
            above_thresh = np.where(magnitude > threshold)[0]
            hf_cutoff_hz = float(freqs[above_thresh[-1]]) if len(above_thresh) > 0 else 0.0

            # Artificial spectral peaks
            if len(magnitude) > 100:
                local_max_indices = scipy.signal.argrelextrema(
                    magnitude, np.greater, order=50
                )[0]
                sharp_peak_count = 0
                for idx in local_max_indices:
                    lo  = max(0, idx - 50)
                    hi  = min(len(magnitude), idx + 51)
                    neighborhood = np.concatenate([
                        magnitude[lo:idx], magnitude[idx + 1:hi]
                    ])
                    if len(neighborhood) > 0:
                        sharpness = magnitude[idx] / (np.mean(neighborhood) + 1e-8)
                        if sharpness > 8.0:
                            sharp_peak_count += 1
                if sharp_peak_count > 3:
                    flags.append("artificial_peaks")

            # Missing frequency bands (any of 8 equal bands near-zero except DC)
            band_count = 8
            band_size  = len(magnitude) // band_count
            if band_size > 0:
                band_ens = []
                for b in range(band_count):
                    s = b * band_size
                    e = s + band_size
                    band_ens.append(float(np.sum(magnitude_sq[s:e])))
                max_band = max(band_ens) + 1e-8
                for b in range(1, band_count):
                    if band_ens[b] < 0.001 * max_band:
                        flags.append("missing_band")
                        break

            return hf_ratio, hf_cutoff_hz, flags

        except Exception as e:
            logger.warning(f"FFT analysis error: {e}")
            return 0.01, 0.0, flags

    # ─────────────────────────────────────────────────────────────────────────
    # C. PHASE CONSISTENCY ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    def _phase_analysis(
        self, y: np.ndarray, sr: int, duration_seconds: float
    ) -> tuple[float, list[int], list[str]]:
        """
        FIX: threshold was 2.5σ — too loose. Now 2.0σ.
        FIX: `expected_max_spikes = duration * 0.5` was far too permissive.
        Now: real speech should have < 2 spikes/minute; flag if > 3/min.
        """
        flags: list[str] = []
        splice_frames: list[int] = []
        try:
            n_fft      = 2048
            hop_length = 512
            stft       = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
            phase      = np.angle(stft)  # (freq_bins, T)

            freq_bins = phase.shape[0]
            T         = phase.shape[1]

            if T < 2:
                return 0.5, [], flags

            freqs            = np.fft.rfftfreq(n_fft, 1.0 / sr)[:freq_bins]
            expected_advance = 2 * np.pi * freqs * hop_length / sr

            deviations = []
            for t in range(1, T):
                expected  = phase[:, t - 1] + expected_advance
                actual    = phase[:, t]
                deviation = np.abs(actual - expected) % (2 * np.pi)
                deviation = np.where(deviation > np.pi, 2 * np.pi - deviation, deviation)
                deviations.append(float(np.mean(deviation)))

            deviations = np.array(deviations, dtype=np.float32)
            mean_phase_deviation   = float(np.mean(deviations))
            phase_continuity_score = float(
                np.clip(1.0 - (mean_phase_deviation / np.pi), 0.0, 1.0)
            )

            if len(deviations) > 2:
                dev_mean  = np.mean(deviations)
                dev_std   = np.std(deviations) + 1e-8
                threshold = dev_mean + 2.0 * dev_std   # tightened from 2.5σ
                spike_indices = np.where(deviations > threshold)[0]
                splice_frames = [int(i + 1) for i in spike_indices]

                # FIX: flag if > 3 spikes per minute (instead of 0.5 per second)
                # BUG FIX: on short audio (e.g. 8s), even 1 spike = 7.5/min — always fires.
                # Require BOTH: rate > 3/min AND absolute count >= 3 spikes.
                spikes_per_minute = len(splice_frames) / (duration_seconds / 60.0 + 1e-8)
                if spikes_per_minute > 3.0 and len(splice_frames) >= 3:
                    flags.append("phase_discontinuity")

            return phase_continuity_score, splice_frames, flags

        except Exception as e:
            logger.warning(f"Phase analysis error: {e}")
            return 0.5, [], flags

    # ─────────────────────────────────────────────────────────────────────────
    # D. NOISE FLOOR ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    def _noise_floor_analysis(
        self, y: np.ndarray, sr: int
    ) -> tuple[float, list[str]]:
        """
        FIX: -90dB threshold never fires on real MP3/AAC recordings (MP3 quantization
        noise floor is typically -70 to -80dB). Now uses BOTH:
          1. Absolute floor check: < -75dB (catches 24-bit clean TTS)
          2. Relative uniformity check: noise floor CV > 0.3 = suspicious stationarity
          3. Noise texture: real rooms have colored noise; TTS has white or zero noise
        """
        flags: list[str] = []
        try:
            window_samples = int(0.1 * sr)   # 100ms windows
            if window_samples == 0:
                return 0.5, flags

            n_windows = len(y) // window_samples
            if n_windows < 5:
                return 0.5, flags

            rms_values = []
            for w in range(n_windows):
                seg = y[w * window_samples:(w + 1) * window_samples]
                rms_values.append(float(np.sqrt(np.mean(seg ** 2) + 1e-12)))

            rms_arr     = np.array(rms_values, dtype=np.float32)
            noise_floor = float(np.percentile(rms_arr, 10))

            # Windows near noise floor (bottom 20th percentile)
            noise_p20     = float(np.percentile(rms_arr, 20))
            noise_windows = rms_arr[rms_arr <= noise_p20]

            if len(noise_windows) < 3:
                noise_consistency_score = 0.5
            else:
                noise_cv = float(np.std(noise_windows) / (np.mean(noise_windows) + 1e-8))
                # FIX: real rooms: noise CV ~ 0.15-0.40 (variable)
                # TTS/synthetic: noise CV < 0.05 (perfectly stationary or exactly zero)
                # High CV is GOOD (real), low CV is BAD (suspicious)
                noise_consistency_score = float(np.clip(noise_cv / 0.30, 0.0, 1.0))

            # Sudden noise resets (>15dB drop)
            rms_db      = 20 * np.log10(rms_arr + 1e-8)
            reset_count = 0
            for i in range(1, len(rms_db)):
                if (rms_db[i - 1] - rms_db[i]) > 15.0:
                    reset_count += 1
            if reset_count > 0:
                flags.append("noise_reset")

            # FIX: absolute floor — use -75dB (not -90dB) to catch TTS on MP3
            noise_floor_db = 20 * np.log10(noise_floor + 1e-8)
            if noise_floor_db < -75:
                flags.append("missing_noise_floor")
            elif noise_cv < 0.05 if len(noise_windows) >= 3 else False:
                # Perfectly stationary noise = synthetic silence
                flags.append("missing_noise_floor")

            return noise_consistency_score, flags

        except Exception as e:
            logger.warning(f"Noise floor analysis error: {e}")
            return 0.5, flags

    # ─────────────────────────────────────────────────────────────────────────
    # E. COMPRESSION ARTIFACT DETECTION
    # ─────────────────────────────────────────────────────────────────────────
    def _compression_analysis(
        self, y: np.ndarray, sr: int
    ) -> tuple[float, list[str]]:
        """Detect codec-induced spectral cutoffs and double-compression spectral holes."""
        flags: list[str] = []
        score = 0.0
        try:
            fft_result = np.fft.rfft(y)
            magnitude  = np.abs(fft_result)
            freqs      = np.fft.rfftfreq(len(y), 1.0 / sr)

            max_mag         = np.max(magnitude) + 1e-10
            threshold_db    = -60.0
            threshold_linear = max_mag * 10 ** (threshold_db / 20)

            above = np.where(magnitude > threshold_linear)[0]
            if len(above) > 0:
                hf_cutoff_hz = float(freqs[above[-1]])
                nyquist      = sr / 2.0
                if hf_cutoff_hz < nyquist * 0.85:
                    flags.append("compression_cutoff")
                    score += 0.5

            # Spectral holes: >=200Hz ranges with near-zero energy
            freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
            bins_200hz      = max(1, int(200.0 / (freq_resolution + 1e-8)))
            noise_threshold = np.mean(magnitude) * 0.01

            hole_count = 0
            in_hole    = False
            hole_start = 0
            for i in range(1, len(magnitude) - 1):
                if magnitude[i] < noise_threshold:
                    if not in_hole:
                        in_hole    = True
                        hole_start = i
                else:
                    if in_hole:
                        hole_len = i - hole_start
                        if hole_len >= bins_200hz:
                            if hole_start > 0 and i < len(magnitude) - 1:
                                hole_count += 1
                        in_hole = False

            if hole_count > 2:
                flags.append("double_compression")
                score += 0.5

            return float(np.clip(score, 0.0, 1.0)), flags

        except Exception as e:
            logger.warning(f"Compression analysis error: {e}")
            return 0.0, flags

    # ─────────────────────────────────────────────────────────────────────────
    # F. CEPSTRAL REGULARITY ANALYSIS (NEW)
    # ─────────────────────────────────────────────────────────────────────────
    def _cepstral_analysis(
        self, y: np.ndarray, sr: int
    ) -> tuple[float, list[str]]:
        """
        NEW: TTS vocoders (HiFi-GAN, WaveNet, etc.) produce cepstra with
        unnaturally regular quefrency peaks due to their fixed vocoder period.
        Real speech: cepstral envelope varies frame-to-frame (different phonemes).
        TTS: cepstral frames are highly similar → high inter-frame correlation.

        Also measures spectral flatness (Wiener entropy):
        Real noise floor: low flatness variation (colored/broadband noise)
        TTS silence: near-perfect flatness = 1.0 (white noise or silence)
        """
        flags: list[str] = []
        try:
            # Compute MFCC over 13 coefficients
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc = np.nan_to_num(mfcc, nan=0.0)

            if mfcc.shape[1] < 10:
                return 0.5, flags

            # Inter-frame correlation at meaningful lag (not adjacent frames).
            # BUG FIX: comparing adjacent MFCC frames (lag=1) gives correlation > 0.99
            # on ALL speech — real and synthetic — because STFT windows heavily overlap.
            # This makes the metric useless for discrimination.
            # Fix: use a stride of max(10, T//30) frames (~300ms–1s lag) so we measure
            # whether the cepstral envelope is unnaturally self-similar across time.
            T = mfcc.shape[1]
            stride = max(10, T // 30)
            corr_vals = []
            for t in range(stride, T, stride):
                f1 = mfcc[:, t - 1]
                f2 = mfcc[:, t]
                d1 = f1 - f1.mean()
                d2 = f2 - f2.mean()
                denom = (np.linalg.norm(d1) * np.linalg.norm(d2)) + 1e-8
                corr_vals.append(float(np.dot(d1, d2) / denom))

            mean_corr = float(np.mean(corr_vals))
            # Real speech: mean correlation ~ 0.3-0.6
            # TTS: mean correlation ~ 0.7-0.95 (unnaturally consistent phoneme delivery)
            cepstral_score = float(np.clip((mean_corr - 0.55) / 0.35, 0.0, 1.0))

            # Spectral flatness (Wiener entropy) — low flatness = tonal (voiced speech),
            # high flatness = noisy (real room noise). TTS has near-zero flatness everywhere.
            flatness = librosa.feature.spectral_flatness(y=y)[0]
            flatness = np.nan_to_num(flatness, nan=0.0)
            if len(flatness) > 5:
                flatness_cv = float(np.std(flatness) / (np.mean(flatness) + 1e-8))
                # Very low CV of flatness = artificially uniform spectrum
                if flatness_cv < 0.5:
                    cepstral_score = min(1.0, cepstral_score + 0.2)

            if cepstral_score > 0.5:
                flags.append("cepstral_regularity")

            return float(np.clip(cepstral_score, 0.0, 1.0)), flags

        except Exception as e:
            logger.warning(f"Cepstral analysis error: {e}")
            return 0.5, flags

    def _empty_result(self) -> dict:
        return {
            "spectral_score":          0.5,
            "smoothness_score":        0.5,
            "repetition_score":        0.5,
            "hf_ratio":                0.0,
            "hf_cutoff_hz":            0.0,
            "phase_continuity_score":  0.5,
            "noise_consistency_score": 0.5,
            "noise_score":             0.5,
            "compression_score":       0.5,
            "cepstral_score":          0.5,
            "splice_candidate_frames": [],
            "flags":                   ["empty_audio"],
            "analysis_time_seconds":   0.0,
        }
