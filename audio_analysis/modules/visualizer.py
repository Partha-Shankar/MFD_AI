"""
visualizer.py — Forensic Report Visualizer

Generates a 4-panel matplotlib figure with:
  1. Mel spectrogram
  2. Pitch (F0) contour with breathing event markers
  3. Energy envelope + ZCR overlay with silence bands
  4. Forensic score summary bar chart with verdict
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import librosa
import librosa.display

logger = logging.getLogger(__name__)


class Visualizer:
    """Generates a PNG forensic report from analysis results."""

    def generate_report(
        self,
        y: np.ndarray,
        sr: int,
        result: dict,
        output_path: str,
    ) -> None:
        """
        Create a 4-panel forensic visualization and save as PNG.

        Args:
            y:           Mono audio waveform.
            sr:          Sample rate in Hz.
            result:      Final verdict dict from VerdictEngine.
            output_path: Where to save the PNG file.
        """
        y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.patch.set_facecolor("#0e1117")
        for ax in axes.flat:
            ax.set_facecolor("#1a1d23")
            ax.tick_params(colors="#cccccc")
            ax.xaxis.label.set_color("#cccccc")
            ax.yaxis.label.set_color("#cccccc")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")

        ax1 = axes[0, 0]
        ax2 = axes[0, 1]
        ax3 = axes[1, 0]
        ax4 = axes[1, 1]

        # ── Panel 1: Mel Spectrogram ──────────────────────────────────────────
        self._plot_mel_spectrogram(ax1, y, sr)

        # ── Panel 2: Pitch Contour + Breathing ───────────────────────────────
        self._plot_pitch_contour(ax2, y, sr)

        # ── Panel 3: Energy + ZCR ─────────────────────────────────────────────
        self._plot_energy_zcr(ax3, y, sr)

        # ── Panel 4: Score Summary ────────────────────────────────────────────
        self._plot_score_summary(ax4, result)

        # Overall title
        verdict = result.get("verdict", "UNKNOWN")
        confidence_pct = int(result.get("confidence", 0) * 100)
        fig.suptitle(
            "DEEPFAKE AUDIO FORENSICS REPORT",
            color="#ffffff",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )
        fig.text(
            0.5, 0.95,
            f"Verdict: {verdict}  |  Confidence: {confidence_pct}%  |  "
            f"Composite Score: {result.get('composite_score', 0):.3f}",
            ha="center",
            color="#aaaaaa",
            fontsize=11,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info(f"Report saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save visualization: {e}")
        finally:
            plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    def _plot_mel_spectrogram(self, ax: plt.Axes, y: np.ndarray, sr: int) -> None:
        try:
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            mel_db = librosa.power_to_db(S + 1e-10, ref=np.max)
            img = librosa.display.specshow(
                mel_db, sr=sr, x_axis="time", y_axis="mel",
                ax=ax, cmap="magma",
            )
            plt.colorbar(img, ax=ax, format="%+2.0f dB").ax.yaxis.set_tick_params(color="#cccccc")
            ax.set_title("Mel Spectrogram", color="#ffffff", fontweight="bold")
            ax.set_xlabel("Time (s)", color="#cccccc")
            ax.set_ylabel("Frequency (Hz)", color="#cccccc")
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes,
                    ha="center", va="center", color="#ff4444")
            ax.set_title("Mel Spectrogram (Error)", color="#ffffff")

    # ─────────────────────────────────────────────────────────────────────────
    def _plot_pitch_contour(self, ax: plt.Axes, y: np.ndarray, sr: int) -> None:
        try:
            import scipy.signal
            f0, voiced_flag, _ = librosa.pyin(
                y,
                fmin=float(librosa.note_to_hz("C2")),
                fmax=float(librosa.note_to_hz("C7")),
                sr=sr,
            )
            f0 = np.nan_to_num(f0, nan=0.0)
            times = librosa.times_like(f0, sr=sr)

            # Shade voiced regions
            voiced_flag = voiced_flag.astype(bool)
            for i in range(1, len(voiced_flag)):
                if voiced_flag[i - 1] and voiced_flag[i]:
                    ax.axvspan(times[i - 1], times[i], alpha=0.15, color="#4488ff")

            # Plot F0
            f0_plot = np.where(voiced_flag, f0, np.nan)
            ax.plot(times, f0_plot, color="#44ccff", linewidth=1.5, label="F0 (Hz)")

            # Breathing event detection (80–600 Hz envelope peaks)
            try:
                nyq = sr / 2.0
                low, high = 80.0 / nyq, min(600.0 / nyq, 0.99)
                if low < high:
                    sos = scipy.signal.butter(4, [low, high], btype="bandpass", output="sos")
                    y_breath = scipy.signal.sosfilt(sos, y.astype(np.float64))
                    analytic = scipy.signal.hilbert(y_breath)
                    envelope = np.abs(analytic).astype(np.float32)
                    smooth_k = max(1, int(0.05 * sr))
                    envelope_smooth = np.convolve(
                        envelope, np.ones(smooth_k) / smooth_k, mode="same"
                    )
                    thresh = np.mean(envelope_smooth) + 1.5 * np.std(envelope_smooth)
                    peaks, _ = scipy.signal.find_peaks(
                        envelope_smooth, height=thresh, distance=max(1, int(0.3 * sr))
                    )
                    breath_times = np.array(peaks) / sr
                    for bt in breath_times:
                        ax.axvline(bt, color="#44ff88", alpha=0.7, linewidth=1.0)
                    if len(breath_times) > 0:
                        ax.plot([], [], color="#44ff88", linewidth=1.0, label="Breath events")
            except Exception:
                pass

            ax.set_title("Pitch Contour + Breathing Events", color="#ffffff", fontweight="bold")
            ax.set_xlabel("Time (s)", color="#cccccc")
            ax.set_ylabel("Frequency (Hz)", color="#cccccc")
            ax.legend(facecolor="#1a1d23", labelcolor="#cccccc", fontsize=8)
            ax.set_ylim(bottom=0)

        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes,
                    ha="center", va="center", color="#ff4444")
            ax.set_title("Pitch Contour (Error)", color="#ffffff")

    # ─────────────────────────────────────────────────────────────────────────
    def _plot_energy_zcr(self, ax: plt.Axes, y: np.ndarray, sr: int) -> None:
        try:
            frame_length = max(1, int(0.020 * sr))
            hop_length = max(1, int(0.010 * sr))
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            rms = np.nan_to_num(rms, nan=0.0)
            rms_times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

            zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)[0]
            zcr = np.nan_to_num(zcr, nan=0.0)
            zcr_times = librosa.times_like(zcr, sr=sr, hop_length=512)

            # Detect silence regions
            silence_threshold = 0.02 * (float(np.max(np.abs(y))) + 1e-8)
            rms_silence_mask = rms < silence_threshold
            in_silence = False
            for i in range(len(rms_silence_mask)):
                if rms_silence_mask[i] and not in_silence:
                    in_silence = True
                    t_start = rms_times[i] if i < len(rms_times) else 0.0
                elif not rms_silence_mask[i] and in_silence:
                    in_silence = False
                    t_end = rms_times[i] if i < len(rms_times) else rms_times[-1]
                    ax.axvspan(t_start, t_end, alpha=0.25, color="#888888")

            # Normalize ZCR to RMS scale for overlay
            rms_max = float(np.max(rms)) + 1e-8
            zcr_max = float(np.max(zcr)) + 1e-8
            zcr_scaled = zcr * (rms_max / zcr_max)

            ax.plot(rms_times, rms, color="#4488ff", linewidth=1.2, label="RMS Energy")
            ax.plot(zcr_times, zcr_scaled, color="#ff8844", linewidth=1.0,
                    alpha=0.8, label="ZCR (scaled)")

            ax.set_title("Energy Envelope & Zero-Crossing Rate", color="#ffffff", fontweight="bold")
            ax.set_xlabel("Time (s)", color="#cccccc")
            ax.set_ylabel("Amplitude", color="#cccccc")
            ax.legend(facecolor="#1a1d23", labelcolor="#cccccc", fontsize=8)

        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes,
                    ha="center", va="center", color="#ff4444")
            ax.set_title("Energy/ZCR (Error)", color="#ffffff")

    # ─────────────────────────────────────────────────────────────────────────
    def _plot_score_summary(self, ax: plt.Axes, result: dict) -> None:
        try:
            scores_dict = result.get("scores", {})
            labels = ["Spectral", "Temporal", "Noise", "Speaker", "Metadata", "Compression"]
            keys = ["spectral", "temporal", "noise", "speaker", "metadata", "compression"]
            values = [float(scores_dict.get(k, 0.5)) for k in keys]

            colors = []
            for v in values:
                if v < 0.4:
                    colors.append("#44cc44")
                elif v < 0.6:
                    colors.append("#cccc44")
                else:
                    colors.append("#cc4444")

            bars = ax.barh(labels, values, color=colors, edgecolor="#333333", height=0.6)

            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(
                    min(val + 0.02, 0.98), bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}",
                    va="center", ha="left",
                    color="#ffffff", fontsize=9,
                )

            # Threshold line
            ax.axvline(0.5, color="#ff4444", linestyle="--", linewidth=1.5,
                       alpha=0.8, label="Suspicion threshold")

            ax.set_xlim(0, 1.0)
            verdict = result.get("verdict", "UNKNOWN")
            confidence_pct = int(result.get("confidence", 0) * 100)
            ax.set_title(
                f"Verdict: {verdict}\n({confidence_pct}% confidence)",
                color="#ffffff", fontweight="bold", fontsize=10,
            )
            ax.set_xlabel("Score (0=authentic, 1=suspicious)", color="#cccccc")

            legend_patches = [
                mpatches.Patch(color="#44cc44", label="Clean (<0.4)"),
                mpatches.Patch(color="#cccc44", label="Uncertain (0.4–0.6)"),
                mpatches.Patch(color="#cc4444", label="Suspicious (>0.6)"),
            ]
            ax.legend(handles=legend_patches, facecolor="#1a1d23",
                      labelcolor="#cccccc", fontsize=8, loc="lower right")

        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes,
                    ha="center", va="center", color="#ff4444")
            ax.set_title("Score Summary (Error)", color="#ffffff")
