import hashlib
import logging
from io import BytesIO

from PIL import Image

from image_analysis.engine.ela import ela_to_base64_overlay, generate_ela
from image_analysis.engine.explainer import generate_explanation
from image_analysis.engine.verdict import compute_verdict
from image_analysis.schemas.response import AnalysisResponse, FlaggedRegion, ModelScores

logger = logging.getLogger(__name__)


class ForensicsPipeline:
    """
    Orchestrates all five forensics models to produce an AnalysisResponse.
    """

    def __init__(self, models: dict) -> None:
        self.ai_detector = models.get("ai_detector")
        self.manipulation = models.get("manipulation")
        self.source_id = models.get("source_id")
        self.patch_localizer = models.get("patch_localizer")
        self.compression = models.get("compression")

    async def analyze(self, image_bytes: bytes) -> AnalysisResponse:
        """
        Run the full forensics pipeline on the provided image bytes.

        Returns a complete AnalysisResponse regardless of model failures.
        """
        image_hash = hashlib.md5(image_bytes).hexdigest()

        try:
            # Step 2: Load image
            image = Image.open(BytesIO(image_bytes)).convert("RGB")

            # Step 3: Generate ELA map
            ela_map = generate_ela(image)

            # Step 4: M1 — AI vs Human
            if self.ai_detector is not None:
                m1_result = self.ai_detector.predict(image)
            else:
                m1_result = {"ai_probability": 0.5, "available": False}

            # Step 5: M2 — Manipulation (ELA-guided)
            if self.manipulation is not None:
                m2_result = self.manipulation.predict(image, ela_map)
            else:
                m2_result = {"manipulation_probability": 0.5, "available": False}

            # Step 6: M3 — Source identification
            if self.source_id is not None:
                m3_result = self.source_id.predict(image)
            else:
                m3_result = {
                    "top_generator": None,
                    "available": False,
                    "stable_diffusion": 0.25,
                    "midjourney": 0.25,
                    "dalle": 0.25,
                }
            source_non_real = max(
                m3_result.get("stable_diffusion", 0.0),
                m3_result.get("midjourney", 0.0),
                m3_result.get("dalle", 0.0),
            )

            # Step 7: M4 — Patch localisation
            if self.patch_localizer is not None:
                m4_result = self.patch_localizer.predict(image)
            else:
                m4_result = {"max_anomaly": 0.5, "flagged_patches": [], "available": False}

            # Step 8: M5 — Compression/upscale
            if self.compression is not None:
                m5_result = self.compression.predict(image)
            else:
                m5_result = {"upscaled": 0.5, "available": False}

            # Step 9: Build scores dict for verdict
            scores_for_verdict = {
                "ai_probability":             m1_result.get("ai_probability", 0.5),
                "manipulation_probability":   m2_result.get("manipulation_probability", 0.5),
                "source_non_real_confidence": source_non_real,
                "patch_max_anomaly":          m4_result.get("max_anomaly", 0.5),
                "upscaled_probability":       m5_result.get("upscaled", 0.5),
            }

            # Step 10: Compute verdict
            verdict_result = compute_verdict(scores_for_verdict)

            # Step 11: Determine generator label
            generator: str | None = m3_result.get("top_generator") or None

            # Step 12: Build flagged regions from M4 patches
            flagged_regions = [
                FlaggedRegion(
                    x=p["x"],
                    y=p["y"],
                    w=p["w"],
                    h=p["h"],
                    label="Anomalous Region",
                    confidence=p["score"],
                )
                for p in m4_result.get("flagged_patches", [])
            ]

            # Step 13: Model availability map
            model_availability = {
                "ai_detector":     m1_result.get("available", True),
                "manipulation":    m2_result.get("available", True),
                "source_id":       m3_result.get("available", True),
                "patch_localizer": m4_result.get("available", True),
                "compression":     m5_result.get("available", True),
            }

            # Step 14: Generate explanation
            explanation = generate_explanation(
                verdict=verdict_result["verdict"],
                scores=scores_for_verdict,
                generator=generator,
                flagged_count=len(flagged_regions),
                model_availability=model_availability,
            )

            # Step 15: ELA overlay as base64
            ela_overlay_b64 = ela_to_base64_overlay(image, ela_map)

            # Step 16: Model scores object
            model_scores = ModelScores(
                ai_detector=m1_result.get("ai_probability", 0.5),
                manipulation=m2_result.get("manipulation_probability", 0.5),
                source_id=source_non_real,
                patch_anomaly=m4_result.get("max_anomaly", 0.5),
                compression=m5_result.get("upscaled", 0.5),
            )

            # Step 17: Return full response
            return AnalysisResponse(
                image_hash=image_hash,
                verdict=verdict_result["verdict"],
                confidence=verdict_result["confidence"],
                scores=model_scores,
                generator=generator,
                flagged_regions=flagged_regions,
                explanation=explanation,
                ela_map_base64=ela_overlay_b64,
            )

        except Exception as exc:
            logger.error("[pipeline] Unhandled exception during analysis: %s", exc)
            return AnalysisResponse(
                image_hash=image_hash,
                verdict="INCONCLUSIVE",
                confidence=0.0,
                scores=ModelScores(
                    ai_detector=0.5,
                    manipulation=0.5,
                    source_id=0.5,
                    patch_anomaly=0.5,
                    compression=0.5,
                ),
                generator=None,
                flagged_regions=[],
                explanation=f"Analysis failed: {str(exc)}",
                ela_map_base64="",
            )
