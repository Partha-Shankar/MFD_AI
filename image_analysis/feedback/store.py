import json
import logging
import os
import threading
from datetime import datetime, timezone

from image_analysis.schemas.request import FeedbackRequest

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def save_feedback(feedback: FeedbackRequest) -> None:
    """
    Append a single feedback record to the JSONL feedback log.
    Thread-safe via a module-level lock.
    """
    path = os.getenv(
        "FEEDBACK_LOG_PATH",
        "./image_analysis/feedback/feedback_log.jsonl",
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    record = {
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "image_hash":       feedback.image_hash,
        "user_verdict":     feedback.user_verdict,
        "original_verdict": feedback.original_verdict,
        "comment":          feedback.comment,
        "agreed":           feedback.user_verdict == feedback.original_verdict,
    }

    with _lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    logger.info("[store] Feedback saved for image_hash=%s", feedback.image_hash)
