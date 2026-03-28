# TODO: Implement fine-tuning trigger when disagreement rate > 20%
# For MVP this file only provides summary statistics.

import json
import logging
import os
from collections import Counter

logger = logging.getLogger(__name__)


def get_feedback_summary() -> dict:
    """
    Read the feedback JSONL log and return aggregate statistics.

    Returns:
        {
            "total": int,
            "agreed": int,
            "disagreed": int,
            "agreement_rate": float,            # 0.0 – 1.0
            "most_common_correction": str | None
        }
    """
    path = os.getenv(
        "FEEDBACK_LOG_PATH",
        "./image_analysis/feedback/feedback_log.jsonl",
    )

    if not os.path.exists(path):
        return {
            "total": 0,
            "agreed": 0,
            "disagreed": 0,
            "agreement_rate": 0.0,
            "most_common_correction": None,
        }

    total = 0
    agreed = 0
    disagreed = 0
    correction_counter: Counter = Counter()

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("[trainer] Skipping malformed feedback line.")
                    continue

                total += 1
                if record.get("agreed", False):
                    agreed += 1
                else:
                    disagreed += 1
                    user_verdict = record.get("user_verdict")
                    if user_verdict:
                        correction_counter[user_verdict] += 1

    except Exception as exc:
        logger.error("[trainer] Failed to read feedback log: %s", exc)
        return {
            "total": 0,
            "agreed": 0,
            "disagreed": 0,
            "agreement_rate": 0.0,
            "most_common_correction": None,
        }

    agreement_rate = (agreed / total) if total > 0 else 0.0
    most_common_correction = (
        correction_counter.most_common(1)[0][0]
        if correction_counter
        else None
    )

    return {
        "total": total,
        "agreed": agreed,
        "disagreed": disagreed,
        "agreement_rate": float(agreement_rate),
        "most_common_correction": most_common_correction,
    }
