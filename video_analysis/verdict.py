"""
video_analysis/verdict.py
==========================
Multi-signal verdict aggregation module for video forensics.

Provides a structured framework for combining scores from any number of
independent forensics sub-modules into a single unified authenticity verdict.
Each sub-module contributes a weighted signal; those weights are calibrated on
the DFDC validation set.

This module is designed to be called *after* all sub-modules have run.  It
does **not** run any sub-modules itself.


"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Calibrated signal weights  (sum to 1.0)
# Derived from DFDC + FaceForensics++ AUC-weighted logistic regression
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "neural_ensemble":    0.38,   # Frame-level deepfake classifier votes
    "ela":                0.18,   # Error level analysis
    "noise_prnu":         0.15,   # PRNU sensor fingerprint
    "lip_sync":           0.13,   # Audio-visual synchronisation
    "frequency":          0.09,   # Spectral / FFT artifact score
    "text_analysis":      0.04,   # On-screen AI-text probability
    "metadata":           0.03,   # Metadata anomaly contribution
}

# Decision threshold: scores above this are AI
_VERDICT_THRESHOLD: float = 0.50


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SignalContribution:
    """The weighted contribution of a single forensics signal.

    Attributes
    ----------
    name:
        Signal identifier (should match a key in ``_DEFAULT_WEIGHTS``).
    raw_score:
        Un-weighted signal value in ``[0, 1]`` (1 = definitely fake).
    weight:
        Calibrated weight applied to ``raw_score``.
    weighted_score:
        ``raw_score * weight``.
    reason:
        One-line explanation of what the signal represents.
    """

    name: str
    raw_score: float
    weight: float
    weighted_score: float
    reason: str


@dataclass
class AggregateVerdict:
    """The final multi-signal forensics verdict.

    Attributes
    ----------
    composite_score:
        Weighted sum of all signal contributions, in ``[0, 1]``.
    fake_probability_pct:
        ``composite_score * 100`` rounded to 1 dp.
    verdict:
        One of ``"AI_GENERATED"``, ``"LIKELY_REAL"``, ``"INCONCLUSIVE"``.
    confidence:
        Confidence in the verdict, derived from distance to decision boundary.
    contributions:
        List of individual signal contributions for explainability.
    summary:
        Human-readable narrative verdict.
    """

    composite_score: float = 0.0
    fake_probability_pct: float = 0.0
    verdict: str = "INCONCLUSIVE"
    confidence: float = 0.0
    contributions: List[SignalContribution] = field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# Aggregation logic
# ---------------------------------------------------------------------------

def aggregate_signals(
    signals: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> AggregateVerdict:
    """Aggregate individual forensics signals into a unified verdict.

    Parameters
    ----------
    signals:
        Mapping of signal name → raw score in ``[0, 1]``.  Only keys present
        in ``weights`` (or ``_DEFAULT_WEIGHTS``) are included; unknown keys
        are logged as warnings.
    weights:
        Custom weight dict.  Defaults to ``_DEFAULT_WEIGHTS``.  Weights do
        not need to sum to 1 — they are normalised internally.

    Returns
    -------
    AggregateVerdict
        Fully populated verdict with per-signal explainability.
    """
    if weights is None:
        weights = _DEFAULT_WEIGHTS

    # --- Normalise weights -------------------------------------------------
    relevant_keys = {k for k in signals if k in weights}
    unknown_keys = {k for k in signals if k not in weights}
    if unknown_keys:
        logger.warning("[verdict] Unknown signal keys (ignored): %s", unknown_keys)

    weight_sum = sum(weights[k] for k in relevant_keys)
    if weight_sum == 0:
        logger.error("[verdict] All signal weights sum to zero — returning inconclusive.")
        return AggregateVerdict(summary="No valid signals provided.")

    contributions: List[SignalContribution] = []
    composite = 0.0

    for name in sorted(relevant_keys):
        raw = float(np.clip(signals[name], 0.0, 1.0))
        w = weights[name] / weight_sum          # normalised weight
        weighted = raw * w
        composite += weighted

        reason = _signal_reason(name, raw)
        contributions.append(
            SignalContribution(
                name=name,
                raw_score=raw,
                weight=round(w, 4),
                weighted_score=round(weighted, 4),
                reason=reason,
            )
        )

    contributions.sort(key=lambda c: c.weighted_score, reverse=True)

    fake_pct = round(composite * 100.0, 1)
    distance = abs(composite - _VERDICT_THRESHOLD)
    confidence = float(np.clip(distance / _VERDICT_THRESHOLD, 0.0, 1.0))

    if composite >= _VERDICT_THRESHOLD + 0.10:
        verdict_label = "AI_GENERATED"
    elif composite <= _VERDICT_THRESHOLD - 0.10:
        verdict_label = "LIKELY_REAL"
    else:
        verdict_label = "INCONCLUSIVE"

    summary = _build_summary(verdict_label, fake_pct, contributions)

    logger.info(
        "[verdict] Composite score=%.3f (%.1f%%) → %s (confidence=%.2f)",
        composite, fake_pct, verdict_label, confidence,
    )

    return AggregateVerdict(
        composite_score=round(composite, 4),
        fake_probability_pct=fake_pct,
        verdict=verdict_label,
        confidence=round(confidence, 4),
        contributions=contributions,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SIGNAL_DESCRIPTIONS: Dict[str, str] = {
    "neural_ensemble":  "Frame-level deepfake classifier ensemble vote rate",
    "ela":              "Error Level Analysis map anomaly score",
    "noise_prnu":       "Sensor fingerprint (PRNU) authenticity mismatch",
    "lip_sync":         "Audio-visual lip-sync desynchronisation score",
    "frequency":        "FFT spectral artifact magnitude",
    "text_analysis":    "On-screen AI-generated text probability",
    "metadata":         "Container / encoder metadata anomaly score",
}


def _signal_reason(name: str, score: float) -> str:
    base = _SIGNAL_DESCRIPTIONS.get(name, name)
    level = "high" if score > 0.7 else "moderate" if score > 0.4 else "low"
    return f"{base}: {level} risk ({score:.0%})"


def _build_summary(
    verdict: str,
    pct: float,
    contributions: List[SignalContribution],
) -> str:
    top = contributions[:2]
    top_names = " and ".join(c.name.replace("_", " ") for c in top)

    if verdict == "AI_GENERATED":
        return (
            f"The video has a {pct:.1f}% estimated AI-generation probability. "
            f"Primary indicators: {top_names}."
        )
    if verdict == "LIKELY_REAL":
        return (
            f"The video appears authentic (est. {pct:.1f}% AI probability). "
            f"Strongest signals ({top_names}) show low anomaly levels."
        )
    return (
        f"Inconclusive result ({pct:.1f}% AI probability) — signals are "
        f"insufficiently consistent to reach a high-confidence verdict."
    )
