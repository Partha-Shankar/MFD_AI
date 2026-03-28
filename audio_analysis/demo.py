"""
demo.py — Demonstration script for the Deepfake Audio Detection System.

Generates two synthetic audio signals:
  - demo_real.wav:  Simulated authentic human speech (jitter, breathing, irregular pauses)
  - demo_fake.wav:  Simulated TTS output (perfect pitch, uniform energy, hard HF cutoff)

Runs both through the full detection pipeline and prints a side-by-side comparison.
"""

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.metadata import MetadataAnalyzer
from modules.spectral import SpectralAnalyzer
from modules.temporal import TemporalAnalyzer
from modules.speaker import SpeakerConsistencyAnalyzer
from modules.fusion import VerdictEngine
from modules.visualizer import Visualizer

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
GREY   = "\033[90m"


def generate_real_audio(output_path: str, sr: int = 22050, duration: float = 8.0) -> np.ndarray:
    """
    Simulate authentic human speech signal:
    - Pitch with biological jitter
    - Natural amplitude modulation
    - Background noise floor
    - Irregular pauses
    """
    np.random.seed(42)
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    # Fundamental frequency with natural variation (jitter + slow modulation)
    f0 = 120.0 + 5.0 * np.sin(2 * np.pi * 0.3 * t) + 2.0 * np.random.randn(n) * 0.1

    # Voiced speech: sum of harmonics with variable amplitude
    voiced = np.zeros(n, dtype=np.float64)
    for harmonic in range(1, 12):
        phase = np.cumsum(2 * np.pi * f0 / sr)
        voiced += np.sin(harmonic * phase) / harmonic

    # Amplitude modulation (natural energy variation)
    envelope = (
        0.5
        + 0.3 * np.sin(2 * np.pi * 0.8 * t)
        + 0.2 * np.abs(np.random.randn(n) * 0.3)
    )
    envelope = np.clip(envelope, 0.05, 1.0)

    # Background noise floor (room acoustics)
    noise = 0.008 * np.random.randn(n)

    # Irregular pauses (natural breath pauses at non-uniform intervals)
    for pause_start_s in [1.5, 3.2, 5.8]:
        pause_start = int(pause_start_s * sr)
        pause_end = int((pause_start_s + 0.3) * sr)
        envelope[pause_start:pause_end] *= 0.05

    # Brief consonant-like bursts
    for burst_start_s in [0.8, 2.1, 4.5, 6.3]:
        bs = int(burst_start_s * sr)
        be = int((burst_start_s + 0.05) * sr)
        if be < n:
            noise[bs:be] += 0.05 * np.random.randn(be - bs)

    real_audio = (voiced * envelope + noise) * 0.3
    real_audio = np.clip(real_audio, -1.0, 1.0).astype(np.float32)
    sf.write(output_path, real_audio, sr)
    return real_audio


def generate_fake_audio(output_path: str, sr: int = 22050, duration: float = 8.0) -> np.ndarray:
    """
    Simulate AI TTS output:
    - Perfectly stable pitch (no jitter)
    - Uniform energy envelope
    - Near-zero noise floor
    - Metronomic pauses at exact intervals
    - Hard cutoff above 8 kHz (codec artifact)
    """
    np.random.seed(123)
    n = int(sr * duration)
    t = np.linspace(0, duration, n, endpoint=False)

    # Perfectly stable fundamental frequency
    f0_stable = 120.0
    phase = 2 * np.pi * f0_stable * t

    # Voiced speech: fewer harmonics, uniform amplitude
    voiced = np.zeros(n, dtype=np.float64)
    for harmonic in range(1, 8):
        voiced += np.sin(harmonic * phase) / harmonic

    # Perfectly uniform energy envelope (no natural variation)
    envelope = 0.5 * np.ones(n, dtype=np.float64)

    # Minimal noise floor (synthetic silence)
    noise = 0.0001 * np.random.randn(n)

    # Uniform pauses at exact 2-second intervals (metronomic TTS behavior)
    for pause_start_s in [2.0, 4.0, 6.0]:
        pause_start = int(pause_start_s * sr)
        pause_end = int((pause_start_s + 0.2) * sr)
        envelope[pause_start:pause_end] = 0.0

    # Hard HF cutoff above 8 kHz (TTS codec)
    combined = voiced * envelope + noise
    fft = np.fft.rfft(combined)
    freqs = np.fft.rfftfreq(len(combined), 1.0 / sr)
    fft[freqs > 8000] = 0.0
    fake_audio = np.fft.irfft(fft)

    # Ensure correct length after irfft
    fake_audio = fake_audio[:n]
    fake_audio = (fake_audio * 0.3).astype(np.float32)
    fake_audio = np.clip(fake_audio, -1.0, 1.0)
    sf.write(output_path, fake_audio, sr)
    return fake_audio


def run_pipeline(y: np.ndarray, sr: int, filepath: str) -> dict:
    """Run the full detection pipeline on a waveform."""
    import librosa as _librosa

    # Resample to 16kHz for analysis
    if sr != 16000:
        y_analysis = _librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr_analysis = 16000
    else:
        y_analysis = y
        sr_analysis = sr

    meta_result     = MetadataAnalyzer().analyze(filepath)
    spectral_result = SpectralAnalyzer().analyze(y_analysis, sr_analysis)
    temporal_result = TemporalAnalyzer().analyze(y_analysis, sr_analysis)
    speaker_result  = SpeakerConsistencyAnalyzer().analyze(y_analysis, sr_analysis)

    file_info = {
        "path": filepath,
        "duration_seconds": round(len(y) / sr, 2),
        "sample_rate": sr,
        "format": Path(filepath).suffix.lstrip(".").upper(),
    }

    return VerdictEngine().decide(
        meta_result, spectral_result, temporal_result, speaker_result,
        file_info=file_info,
    )


def print_comparison(real_result: dict, fake_result: dict) -> None:
    """Print a formatted side-by-side comparison table."""
    real_verdict = real_result.get("verdict", "UNKNOWN")
    fake_verdict = fake_result.get("verdict", "UNKNOWN")
    real_conf = int(real_result.get("confidence", 0) * 100)
    fake_conf = int(fake_result.get("confidence", 0) * 100)
    real_flags = len(real_result.get("flags", []))
    fake_flags = len(fake_result.get("flags", []))
    real_score = real_result.get("composite_score", 0.0)
    fake_score = fake_result.get("composite_score", 0.0)

    real_color = GREEN if "AUTHENTIC" in real_verdict else RED
    fake_color = RED if "GENERATED" in fake_verdict or "CLONED" in fake_verdict else YELLOW

    col_w = 38
    print("\n" + "=" * (col_w * 2 + 5))
    print(f"  {BOLD}DEEPFAKE AUDIO DETECTOR — DEMO RESULTS{RESET}")
    print("=" * (col_w * 2 + 5))

    hdr = f"{'REAL AUDIO':<{col_w}}  {'FAKE AUDIO':<{col_w}}"
    print(f"  {BOLD}{hdr}{RESET}")
    print("-" * (col_w * 2 + 5))

    def row(label: str, real_val: str, fake_val: str,
            real_c: str = WHITE, fake_c: str = WHITE) -> None:
        print(
            f"  {label:<14}  {real_c}{real_val:<{col_w - 16}}{RESET}  "
            f"{fake_c}{fake_val:<{col_w - 16}}{RESET}"
        )

    row("Verdict:",   real_verdict[:col_w-16], fake_verdict[:col_w-16],
        real_color, fake_color)
    row("Confidence:", f"{real_conf}%", f"{fake_conf}%")
    row("Score:",      f"{real_score:.4f}", f"{fake_score:.4f}")
    row("Flags:",      str(real_flags), str(fake_flags))

    # Per-module scores
    print("-" * (col_w * 2 + 5))
    for key, label in [
        ("spectral", "Spectral"), ("temporal", "Temporal"),
        ("noise", "Noise"), ("metadata", "Metadata"),
    ]:
        rs = real_result.get("scores", {}).get(key, 0.0)
        fs = fake_result.get("scores", {}).get(key, 0.0)
        rc = GREEN if rs < 0.4 else (YELLOW if rs < 0.6 else RED)
        fc = GREEN if fs < 0.4 else (YELLOW if fs < 0.6 else RED)
        row(f"{label}:", f"{rs:.3f}", f"{fs:.3f}", rc, fc)

    print("=" * (col_w * 2 + 5))

    # Anomalies
    real_anomalies = real_result.get("anomalies", [])
    fake_anomalies = fake_result.get("anomalies", [])

    if real_anomalies:
        print(f"\n  {BOLD}Real audio anomalies:{RESET}")
        for a in real_anomalies[:5]:
            print(f"    {YELLOW}⚠{RESET}  {a}")

    if fake_anomalies:
        print(f"\n  {BOLD}Fake audio anomalies:{RESET}")
        for a in fake_anomalies[:8]:
            print(f"    {RED}⚠{RESET}  {a}")

    print()


def main() -> None:
    sr = 22050

    print(f"\n{CYAN}{BOLD}DEEPFAKE AUDIO DETECTOR — DEMO{RESET}")
    print(f"{GREY}Generating synthetic test signals...{RESET}\n")

    # Generate test signals
    real_path = "demo_real.wav"
    fake_path = "demo_fake.wav"

    print(f"{GREY}  → Generating simulated authentic speech...{RESET}")
    real_audio = generate_real_audio(real_path, sr=sr, duration=8.0)
    print(f"{GREY}  → Generating simulated TTS speech...{RESET}")
    fake_audio = generate_fake_audio(fake_path, sr=sr, duration=8.0)
    print(f"{GREEN}  ✓ Signals written: {real_path}, {fake_path}{RESET}\n")

    # Run pipeline on both
    print(f"{CYAN}Analyzing real audio...{RESET}")
    t0 = time.time()
    real_result = run_pipeline(real_audio, sr, real_path)
    real_time = time.time() - t0

    print(f"{CYAN}Analyzing fake audio...{RESET}")
    t0 = time.time()
    fake_result = run_pipeline(fake_audio, sr, fake_path)
    fake_time = time.time() - t0

    print(f"{GREY}Analysis times: real={real_time:.2f}s, fake={fake_time:.2f}s{RESET}")

    # Print comparison
    print_comparison(real_result, fake_result)

    # Generate visualizations
    print(f"{CYAN}Generating forensic report images...{RESET}")
    viz = Visualizer()
    try:
        import librosa as _librosa
        real_16k = _librosa.resample(real_audio, orig_sr=sr, target_sr=16000)
        fake_16k = _librosa.resample(fake_audio, orig_sr=sr, target_sr=16000)

        viz.generate_report(real_16k, 16000, real_result, "demo_real_report.png")
        print(f"{GREEN}  ✓ Saved: demo_real_report.png{RESET}")

        viz.generate_report(fake_16k, 16000, fake_result, "demo_fake_report.png")
        print(f"{GREEN}  ✓ Saved: demo_fake_report.png{RESET}")
    except Exception as e:
        print(f"{YELLOW}  ⚠ Visualization failed: {e}{RESET}")

    print(f"\n{GREEN}{BOLD}Demo complete.{RESET}\n")


if __name__ == "__main__":
    main()



#python -m uvicorn api:app --port 8000