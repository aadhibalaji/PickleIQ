"""FastAPI entry point. Run with:
    uvicorn app.main:app --port 8420
"""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import db
from .api import analyze, annotate, chat, media, models, rally_analysis
from .config import RUN_PROCESSING, STORAGE_BACKEND
from .jobs import runner
from .learning import params as learned_params

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(title="Pickleball Rally Detection")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    db.init_db()
    learned_params.ensure_default_version()
    # Jobs that were queued/running when the previous process died can never
    # finish — mark them so the UI offers a clean retry instead of a stuck bar.
    with db.connect() as con:
        con.execute(
            "UPDATE jobs SET status='error', message='server restarted mid-job',"
            " updated_at=? WHERE status IN ('queued','running')", (db.now(),))
    runner.start()
    # EC2-worker-only: discovers videos a RUN_PROCESSING=0 UI-only container
    # uploaded to S3 and starts processing them on THIS instance — see
    # cloud_ingest.py for why this is needed (the S3-upload Lambda trigger
    # only powers the instance on; nothing else ever told it which video).
    # Never runs on a UI-only container (RUN_PROCESSING=0) — that container
    # has no business discovering/starting CV work, only uploading/syncing.
    if RUN_PROCESSING and STORAGE_BACKEND == "s3":
        from . import cloud_ingest
        cloud_ingest.start()


app.include_router(media.router, prefix="/api")
app.include_router(annotate.router, prefix="/api")
app.include_router(analyze.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(rally_analysis.router, prefix="/api")


@app.get("/api/health")
def health():
    return {"ok": True}


# Serve the built frontend when it exists (production mode); during
# development the Vite dev server proxies /api here instead.
_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="frontend")
