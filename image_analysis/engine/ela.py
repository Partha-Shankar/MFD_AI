import base64
import io
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def generate_ela(image: Image.Image, quality: int = 95) -> Image.Image:
    """
    Generate an Error Level Analysis (ELA) map for the given image.

    Args:
        image:   Source PIL image (any mode).
        quality: JPEG re-save quality level (default 95).

    Returns:
        PIL Image representing the amplified pixel difference (ELA map).
        On error, returns a black image of the same size.
    """
    try:
        original_rgb = image.convert("RGB")

        # Re-save at reduced quality to a buffer
        buffer = io.BytesIO()
        original_rgb.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        resaved = Image.open(buffer).convert("RGB")

        # Compute amplified absolute difference
        original_arr = np.array(original_rgb).astype(np.int16)
        resaved_arr = np.array(resaved).astype(np.int16)
        ela_arr = np.abs(original_arr - resaved_arr) * 20
        ela_arr = np.clip(ela_arr, 0, 255).astype(np.uint8)

        return Image.fromarray(ela_arr)
    except Exception as exc:
        logger.error("[ela] generate_ela failed: %s", exc)
        # Return a black image of the same size as the input
        try:
            w, h = image.size
        except Exception:
            w, h = 224, 224
        return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))


def ela_to_base64_overlay(
    original: Image.Image, ela_map: Image.Image
) -> str:
    """
    Blend the ELA map over the original image and return a base64-encoded PNG.

    Args:
        original: Original PIL image.
        ela_map:  ELA PIL image (same or different size — will be resized).

    Returns:
        Base64-encoded PNG string.
    """
    try:
        # Ensure ela_map matches original dimensions
        if ela_map.size != original.size:
            ela_map = ela_map.resize(original.size, Image.LANCZOS)

        blended = Image.blend(
            original.convert("RGBA"), ela_map.convert("RGBA"), alpha=0.5
        )

        out_buffer = io.BytesIO()
        blended.save(out_buffer, format="PNG")
        return base64.b64encode(out_buffer.getvalue()).decode("utf-8")
    except Exception as exc:
        logger.error("[ela] ela_to_base64_overlay failed: %s", exc)
        return ""
