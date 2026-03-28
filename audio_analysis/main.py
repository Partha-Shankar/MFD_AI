#!/usr/bin/env python3
"""
main.py — CLI entry point for the Deepfake Audio Detection System.

Usage:
    python main.py audio.wav
    python main.py audio.mp3 --verbose
    python main.py audio.flac --output report.json --visualize
    python main.py long_speech.wav --sliding --window-size 30
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import soundfile as sf

from modules.metadata import MetadataAnalyzer
from modules.spectral import SpectralAnalyzer
from modules.temporal import TemporalAnalyzer
from modules.speaker import SpeakerConsistencyAnalyzer
from modules.fusion import VerdictEngine
from modules.visualizer import Visualizer

# ── Terminal colours ──────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
GREY   = "\033[90m"

VERDICT_COLORS = {
    "AUTHENTIC HUMAN SPEECH":       GREEN,
    "AI GENERATED SPEECH (TTS)":    RED,
    "VOICE CLONED SPEECH":          RED,
    "EDITED / SPLICED AUDIO":       YELLOW,
    "AUTO TUNED":                   WHITE,
}

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("main")

TARGET_SR = 16000  # Normalize to 16 kHz for consistent analysis
MAX_WINDOW_SECONDS = 300  # Auto-window audio > 5 minutes

# Music energy threshold: if the repeating (music) component holds more than
# 35% of total signal energy, route speaker/temporal analysis to vocal-isolated audio.
MUSIC_ENERGY_THRESHOLD = 0.35


def separate_vocals(y: np.ndarray) -> np.ndarray:
    """
    Vocal isolation via REPET (REpeating Pattern Extraction Technique).
    Uses librosa's nn_filter with median aggregation to estimate the repeating
    background (music), then soft-masks it out to leave speech.

    Returns the vocal-isolated waveform (same length as y).
    Falls back to y on any error.
    """
    try:
        S_full, phase = librosa.magphase(librosa.stft(y))
        S_background  = librosa.decompose.nn_filter(
            S_full, aggregate=np.median, metric="cosine"
        )
        S_background  = np.minimum(S_full, S_background)
        # margin=2: foreground (vocal) must be 2× louder than estimated background
        mask_vocal    = librosa.util.softmask(
            S_full - S_background, 2 * S_background, power=2
        )
        y_vocal = librosa.istft(mask_vocal * S_full * phase, length=len(y))
        return y_vocal.astype(np.float32)
    except Exception as e:
        logger.warning(f"Vocal separation failed, using original: {e}")
        return y


def detect_music_contamination(y: np.ndarray) -> float:
    """
    Returns the fraction of signal energy in the repeating (music) component.
    Values > MUSIC_ENERGY_THRESHOLD trigger vocal separation pre-pass.
    """
    try:
        S_full, _     = librosa.magphase(librosa.stft(y))
        S_background  = librosa.decompose.nn_filter(
            S_full, aggregate=np.median, metric="cosine"
        )
        S_background  = np.minimum(S_full, S_background)
        music_energy  = float(np.sum(S_background ** 2))
        total_energy  = float(np.sum(S_full ** 2)) + 1e-8
        return music_energy / total_energy
    except Exception:
        return 0.0


def run_pipeline(y: np.ndarray, sr: int, filepath: str) -> dict:
    """Run all four analysis modules and produce a verdict."""
    meta_analyzer     = MetadataAnalyzer()
    spectral_analyzer = SpectralAnalyzer()
    temporal_analyzer = TemporalAnalyzer()
    speaker_analyzer  = SpeakerConsistencyAnalyzer()
    engine            = VerdictEngine()

    t0 = time.time()

    # ── Vocal separation pre-pass ─────────────────────────────────────────────
    # Background music confounds temporal (F0 jitter), speaker (embeddings),
    # and noise-floor analysis by adding realistic-looking signal variation.
    # When significant music energy is detected, run music-sensitive modules
    # on the vocal-isolated signal instead.
    music_fraction = detect_music_contamination(y)
    y_for_temporal = y
    y_for_speaker  = y
    music_detected = music_fraction > MUSIC_ENERGY_THRESHOLD
    if music_detected:
        print(f"{GREY}[Pre-pass] Music/background detected ({music_fraction:.0%} energy) "
              f"— isolating vocals for temporal & speaker analysis{RESET}")
        y_vocal        = separate_vocals(y)
        y_for_temporal = y_vocal
        y_for_speaker  = y_vocal

    metadata_result = meta_analyzer.analyze(filepath)
    spectral_result = spectral_analyzer.analyze(y, sr)          # always on full mix
    temporal_result = temporal_analyzer.analyze(y_for_temporal, sr)
    speaker_result  = speaker_analyzer.analyze(y_for_speaker, sr)

    if music_detected:
        # Tag the result so the explanation can mention it
        temporal_result["_music_separated"] = True
        speaker_result["_music_separated"]  = True

    info = sf.info(filepath)
    file_info = {
        "path":             filepath,
        "duration_seconds": round(float(len(y) / sr), 2),
        "sample_rate":      sr,
        "format":           Path(filepath).suffix.lstrip(".").upper(),
        "file_size_bytes":  Path(filepath).stat().st_size,
    }

    result = engine.decide(
        metadata_result, spectral_result, temporal_result, speaker_result,
        file_info=file_info,
    )
    result["total_analysis_time_seconds"] = round(time.time() - t0, 3)
    result["_module_times"] = {
        "metadata": metadata_result.get("analysis_time_seconds", 0),
        "spectral":  spectral_result.get("analysis_time_seconds", 0),
        "temporal":  temporal_result.get("analysis_time_seconds", 0),
        "speaker":   speaker_result.get("analysis_time_seconds", 0),
    }
    result["_module_results"] = {
        "metadata": metadata_result,
        "spectral": spectral_result,
        "temporal": temporal_result,
        "speaker":  speaker_result,
    }
    return result


def sliding_window_analysis(
    y: np.ndarray, sr: int, filepath: str, window_size: int
) -> dict:
    """
    Analyze long audio in overlapping windows and produce a majority-vote verdict.
    """
    window_samples = window_size * sr
    hop_samples = window_samples // 2
    total_samples = len(y)

    if total_samples <= window_samples:
        return run_pipeline(y, sr, filepath)

    windows = []
    pos = 0
    while pos + window_samples <= total_samples:
        windows.append((pos, y[pos:pos + window_samples]))
        pos += hop_samples

    print(f"\n{CYAN}[Sliding window mode] Analyzing {len(windows)} windows "
          f"({window_size}s each)...{RESET}")

    window_results = []
    for i, (offset, w_audio) in enumerate(windows):
        t_start = offset / sr
        t_end = (offset + window_samples) / sr
        sys.stdout.write(f"\r  Window {i+1}/{len(windows)}: {t_start:.0f}s–{t_end:.0f}s")
        sys.stdout.flush()
        r = run_pipeline(w_audio, sr, filepath)
        r["window_start_seconds"] = round(t_start, 2)
        r["window_end_seconds"] = round(t_end, 2)
        window_results.append(r)
    print()

    # Majority vote weighted by confidence
    verdict_votes: dict[str, float] = {}
    for r in window_results:
        v = r["verdict"]
        c = r.get("confidence", 0.5)
        verdict_votes[v] = verdict_votes.get(v, 0.0) + c

    final_verdict = max(verdict_votes, key=lambda k: verdict_votes[k])
    mean_confidence = float(np.mean([r.get("confidence", 0.5) for r in window_results]))
    mean_composite  = float(np.mean([r.get("composite_score", 0.5) for r in window_results]))

    all_flags: set[str] = set()
    for r in window_results:
        all_flags.update(r.get("flags", []))

    return {
        "verdict": final_verdict,
        "confidence": round(mean_confidence, 4),
        "composite_score": round(mean_composite, 4),
        "scores": window_results[-1].get("scores", {}),
        "anomalies": window_results[-1].get("anomalies", []),
        "flags": sorted(list(all_flags)),
        "explanation": (
            f"Sliding window analysis over {len(windows)} segments. "
            f"Majority verdict: {final_verdict} (confidence: {int(mean_confidence*100)}%)."
        ),
        "window_results": window_results,
        "file_info": window_results[0].get("file_info", {}),
    }


def print_result(result: dict, verbose: bool = False) -> None:
    """Pretty-print the verdict to the terminal."""
    verdict = result.get("verdict", "UNKNOWN")
    confidence = result.get("confidence", 0.0)
    composite = result.get("composite_score", 0.0)
    anomalies = result.get("anomalies", [])
    flags = result.get("flags", [])
    explanation = result.get("explanation", "")

    color = VERDICT_COLORS.get(verdict, WHITE)

    print("\n" + "=" * 60)
    print(f"{BOLD}  DEEPFAKE AUDIO FORENSICS RESULT{RESET}")
    print("=" * 60)
    print(f"  Verdict:    {color}{BOLD}{verdict}{RESET}")
    print(f"  Confidence: {color}{int(confidence * 100)}%{RESET}")
    print(f"  Score:      {composite:.4f}  (0=authentic, 1=fake)")
    print("-" * 60)

    if anomalies:
        print(f"\n  {BOLD}Anomalies Detected ({len(anomalies)}):{RESET}")
        for a in anomalies:
            print(f"    {YELLOW}⚠{RESET}  {a}")
    else:
        print(f"\n  {GREEN}✓  No forensic anomalies detected{RESET}")

    print(f"\n  {BOLD}Explanation:{RESET}")
    # Wrap explanation at 70 chars
    words = explanation.split()
    line = "  "
    for word in words:
        if len(line) + len(word) + 1 > 72:
            print(line)
            line = "    " + word
        else:
            line += (" " if line.strip() else "") + word
    if line.strip():
        print(line)

    if verbose:
        print(f"\n  {BOLD}Per-Module Scores:{RESET}")
        scores = result.get("scores", {})
        for module, score in scores.items():
            bar_len = int(score * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            sc = float(score)
            bar_color = GREEN if sc < 0.4 else (YELLOW if sc < 0.6 else RED)
            print(f"    {module:<12}  {bar_color}{bar}{RESET}  {sc:.3f}")

        module_results = result.get("_module_results", {})
        if module_results:
            print(f"\n  {BOLD}Temporal Details:{RESET}")
            t = module_results.get("temporal", {})
            print(f"    Breath rate:    {t.get('breath_rate_per_minute', 0):.1f} /min")
            print(f"    F0 jitter:      {t.get('f0_jitter', 0):.5f}")
            print(f"    F0 range:       {t.get('f0_range_hz', 0):.1f} Hz")
            print(f"    Voiced ratio:   {t.get('voiced_ratio', 0):.3f}")
            print(f"    Pause count:    {t.get('pause_count', 0)}")

            sp = module_results.get("speaker", {})
            print(f"\n  {BOLD}Speaker Details:{RESET}")
            print(f"    Segments:       {sp.get('segment_count', 0)}")
            print(f"    Mean similarity:{sp.get('mean_segment_similarity', 1.0):.4f}")
            note = sp.get("note", "")
            if note:
                print(f"    Note:           {note}")

        times = result.get("_module_times", {})
        if times:
            total = result.get("total_analysis_time_seconds", 0)
            print(f"\n  {BOLD}Timing:{RESET}")
            for mod, t in times.items():
                print(f"    {mod:<12}  {t:.3f}s")
            print(f"    {'total':<12}  {total:.3f}s")

    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deepfake Audio Detection System — signal forensics, no ML training required."
    )
    parser.add_argument("filepath", help="Path to the audio file to analyze")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed per-module scores")
    parser.add_argument("--output", "-o", default=None,
                        help="Save JSON result to this file")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate a PNG forensic report visualization")
    parser.add_argument("--sliding", action="store_true",
                        help="Use sliding-window analysis for long audio")
    parser.add_argument("--window-size", type=int, default=30,
                        help="Window size in seconds for sliding mode (default: 30)")
    parser.add_argument("--log-level", default="WARNING",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Set logging verbosity")
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    filepath = args.filepath
    if not Path(filepath).exists():
        print(f"{RED}Error: File not found: {filepath}{RESET}")
        sys.exit(1)

    print(f"\n{CYAN}Loading audio: {filepath}{RESET}")

    try:
        y_raw, sr_raw = librosa.load(filepath, sr=None, mono=True)
    except Exception as e:
        print(f"{RED}Error loading audio: {e}{RESET}")
        sys.exit(1)

    # Resample to TARGET_SR for consistent analysis
    if sr_raw != TARGET_SR:
        print(f"{GREY}Resampling from {sr_raw} Hz → {TARGET_SR} Hz{RESET}")
        y = librosa.resample(y_raw, orig_sr=sr_raw, target_sr=TARGET_SR)
        sr = TARGET_SR
    else:
        y = y_raw
        sr = sr_raw

    duration = len(y) / sr
    print(f"{GREY}Duration: {duration:.1f}s  |  SR: {sr} Hz  |  Samples: {len(y):,}{RESET}")

    # Auto-window very long audio
    if duration > MAX_WINDOW_SECONDS and not args.sliding:
        print(f"{YELLOW}Audio >5 minutes — auto-enabling sliding window mode{RESET}")
        args.sliding = True

    # Run analysis
    if args.sliding:
        result = sliding_window_analysis(y, sr, filepath, args.window_size)
    else:
        result = run_pipeline(y, sr, filepath)

    print_result(result, verbose=args.verbose)

    # Save JSON output
    if args.output:
        # Remove non-serializable internal keys for clean output
        output_result = {k: v for k, v in result.items() if not k.startswith("_")}
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(output_result, f, indent=2, default=str)
            print(f"{GREEN}JSON report saved: {args.output}{RESET}\n")
        except Exception as e:
            print(f"{RED}Failed to save JSON: {e}{RESET}")

    # Generate visualization
    if args.visualize:
        viz_path = str(Path(filepath).stem) + "_forensic_report.png"
        print(f"{CYAN}Generating visualization → {viz_path}{RESET}")
        try:
            viz = Visualizer()
            viz.generate_report(y, sr, result, viz_path)
            print(f"{GREEN}Visualization saved: {viz_path}{RESET}\n")
        except Exception as e:
            print(f"{RED}Visualization failed: {e}{RESET}")


if __name__ == "__main__":
    main()
