def compute_verdict(scores: dict) -> dict:
    """
    Compute a verdict from the five model score signals.

    Expected keys:
        "ai_probability"             float 0–1  (M1)
        "manipulation_probability"   float 0–1  (M2)
        "source_non_real_confidence" float 0–1  (M3)
        "patch_max_anomaly"          float 0–1  (M4)
        "upscaled_probability"       float 0–1  (M5)

    Returns:
        { "verdict": str, "confidence": float, "composite_score": float }
    """
    ai_prob = float(scores.get("ai_probability", 0.5))
    manip_prob = float(scores.get("manipulation_probability", 0.5))
    source_conf = float(scores.get("source_non_real_confidence", 0.5))
    patch_anomaly = float(scores.get("patch_max_anomaly", 0.5))
    upscaled_prob = float(scores.get("upscaled_probability", 0.5))

    composite = (
        ai_prob       * 0.35
        + manip_prob  * 0.25
        + patch_anomaly * 0.20
        + source_conf * 0.10
        + upscaled_prob * 0.10
    )

    # Decision tree — first match wins
    if composite > 0.65:
        verdict = "AI Image"
        confidence = composite
    elif manip_prob > 0.55 or patch_anomaly > 0.60 or upscaled_prob > 0.75:
        verdict = "AI Manipulated Image"
        confidence = max(manip_prob, patch_anomaly, upscaled_prob)
    else:
        verdict = "Real Image"
        confidence = 1.0 - composite

    return {
        "verdict": verdict,
        "confidence": float(confidence),
        "composite_score": float(composite),
    }
