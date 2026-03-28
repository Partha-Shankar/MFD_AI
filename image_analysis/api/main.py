import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Force HuggingFace to download models to the D drive
os.environ["HF_HOME"] = r"D:\huggingface_cache"

load_dotenv()

# --- Logging Configuration ---
LOG_FILE = "system_backend.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from image_analysis.engine.pipeline import ForensicsPipeline
    from image_analysis.models.loader import load_all_models

    model_dir = os.getenv("LOCAL_MODEL_DIR", "./models")
    models = load_all_models(model_dir)
    app.state.pipeline = ForensicsPipeline(models)
    loaded = sum(1 for v in models.values() if v is not None)
    logger.info(f"[image_analysis] {loaded}/5 models loaded")
    print(f"[image_analysis] {loaded}/5 models loaded")
    yield
    # Cleanup on shutdown (nothing required)


app = FastAPI(title="MFD Image Analysis API", lifespan=lifespan)

origins = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from image_analysis.api.feedback import feedback_router  # noqa: E402
from image_analysis.api.router import router  # noqa: E402

app.include_router(router)
app.include_router(feedback_router)


@app.get("/health")
async def health():
    pipeline = app.state.pipeline
    model_status = {
        "ai_detector":     pipeline.ai_detector is not None,
        "manipulation":    pipeline.manipulation is not None,
        "source_id":       pipeline.source_id is not None,
        "patch_localizer": pipeline.patch_localizer is not None,
        "compression":     pipeline.compression is not None,
    }
    return {
        "status": "ok",
        "models_loaded": sum(model_status.values()),
        "model_status": model_status,
    }


@app.get("/system-logs")
async def get_system_logs():
    """Returns the last 500 lines of the system_backend.log file."""
    if not os.path.exists(LOG_FILE):
        return JSONResponse({"logs": []})
    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
        # Return the last 500 lines
        return JSONResponse({"logs": lines[-500:]})
    except Exception as e:
        logger.error(f"Failed to read logs: {e}")
        return JSONResponse({"logs": [f"Error reading logs: {e}"]})
