from pydantic import BaseModel


class FlaggedRegion(BaseModel):
    x: int
    y: int
    w: int
    h: int
    label: str
    confidence: float


class ModelScores(BaseModel):
    ai_detector: float        # M1 — probability image is AI
    manipulation: float       # M2 — probability of manipulation
    source_id: float          # M3 — confidence in top generator
    patch_anomaly: float      # M4 — max anomaly across patches
    compression: float        # M5 — upscale/compression probability


class AnalysisResponse(BaseModel):
    image_hash: str
    verdict: str
    # Possible values:
    # "AI_GENERATED" | "AI_EDITED" | "AUTHENTIC" |
    # "COMPOSITED" | "UPSCALED" | "INCONCLUSIVE"

    confidence: float         # 0.0 to 1.0

    scores: ModelScores

    generator: str | None     # "Midjourney" | "SDXL" | "DALL-E" |
                              # "GAN" | "Unknown" | None

    flagged_regions: list[FlaggedRegion]

    explanation: str          # Human-readable verdict explanation

    ela_map_base64: str       # Base64 PNG of ELA heatmap overlay
