def generate_explanation(
    verdict: str,
    scores: dict,
    generator: str | None,
    flagged_count: int,
    model_availability: dict,
) -> str:
    """
    Generate a human-readable explanation paragraph for a forensics verdict.

    Rules:
    - NO external API calls. NO LLM calls. Pure logic + f-strings.
    - 3–5 sentences, plain English, no technical jargon.
    - Only mentions signals from models that returned available=True.

    Args:
        verdict:            One of the six verdict strings.
        scores:             Dict with the five probability scores.
        generator:          AI generator name or None.
        flagged_count:      Number of anomalous patches detected.
        model_availability: Which models returned available=True.
    """

    ai_prob = float(scores.get("ai_probability", 0.5))
    manip_prob = float(scores.get("manipulation_probability", 0.5))
    source_conf = float(scores.get("source_non_real_confidence", 0.5))
    patch_val = float(scores.get("patch_max_anomaly", 0.5))
    upscaled_prob = float(scores.get("upscaled_probability", 0.5))

    available_ai = model_availability.get("ai_detector", True)
    available_manip = model_availability.get("manipulation", True)
    available_source = model_availability.get("source_id", True)
    available_patch = model_availability.get("patch_localizer", True)
    available_comp = model_availability.get("compression", True)

    confidence_pct = _pct(
        ai_prob if verdict == "AI Image"
        else manip_prob if verdict == "AI Manipulated Image"
        else 1.0 - (
            ai_prob * 0.35 + manip_prob * 0.25 + patch_val * 0.20
            + source_conf * 0.10 + upscaled_prob * 0.10
        ) if verdict == "Real Image"
        else 50
    )

    # Sentence 1 — verdict summary
    verdict_labels = {
        "AI Image": "AI-generated",
        "AI Manipulated Image": "AI-manipulated",
        "Real Image": "authentic",
    }
    label = verdict_labels.get(verdict, verdict.lower())
    s1 = f"This image was identified as {label} with {confidence_pct}% confidence."

    # Collect available signals ordered by strength
    signals: list[tuple[float, str]] = []
    if available_ai:
        signals.append((ai_prob, f"the AI detection model scored the image {_pct(ai_prob)}% likely to be synthetic"))
    if available_manip:
        signals.append((manip_prob, f"the manipulation analysis model found {_pct(manip_prob)}% evidence of pixel-level editing"))
    if available_source and source_conf > 0.1:
        signals.append((source_conf, f"the generator attribution model detected a {_pct(source_conf)}% match with known AI styles"))
    if available_patch:
        signals.append((patch_val, f"the region scan flagged a peak anomaly score of {_pct(patch_val)}%"))
    if available_comp:
        signals.append((upscaled_prob, f"the compression pattern analysis flagged this image {_pct(upscaled_prob)}% likely to have been artificially scaled"))

    signals.sort(key=lambda x: x[0], reverse=True)

    # Sentence 2 — strongest signal
    s2 = ""
    if signals:
        s2 = f"The strongest signal came from {signals[0][1]}."

    # Sentence 3 — second strongest OR generator name
    s3 = ""
    if len(signals) >= 2 and signals[1][0] > 0.3:
        s3 = f"Supporting evidence: {signals[1][1]}."
    if available_source and generator:
        s3 = f"The generator attribution model identified the style as most consistent with {generator} ({_pct(source_conf)}% match)."

    # Sentence 4 — flagged regions (if any)
    s4 = ""
    if flagged_count > 0 and available_patch:
        region_word = "region" if flagged_count == 1 else "regions"
        s4 = (
            f"Additionally, {flagged_count} image {region_word} showed localized "
            "irregularities typical of AI synthesis or digital manipulation."
        )

    # Sentence 5 — authentic reassurance
    s5 = ""
    if verdict == "Real Image":
        checks: list[str] = []
        if available_ai:
            checks.append("AI pattern detection")
        if available_manip:
            checks.append("manipulation analysis")
        if available_comp:
            checks.append("compression pattern analysis")
        if checks:
            s5 = f"All available checks ({', '.join(checks)}) returned results consistent with an unmodified photograph."

    parts = [p for p in [s1, s2, s3, s4, s5] if p]
    return " ".join(parts)


def _pct(value: float) -> int:
    """Convert a 0–1 float to an integer percentage."""
    return int(round(float(value) * 100))
