"""Trained-model registry endpoints: list versions, promote/reject, and the
learning curve (held-out boundary error vs accumulated labels)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..learning import model_registry

router = APIRouter()


@router.get("/models")
def list_models(kind: str | None = None):
    return model_registry.history(kind)


@router.get("/models/learning_curve")
def learning_curve(kind: str = "rally_end"):
    """Held-out end/start boundary error per model version against the number of
    labeled windows it trained on — the literal 'gets better as labels
    accumulate' chart. Empty until a model has been validated (>= 3 clips)."""
    curve = []
    for m in sorted(model_registry.history(kind), key=lambda r: r["version"]):
        after = (m.get("metrics") or {}).get("after") or {}
        trained = m.get("trained_on") or {}
        curve.append({
            "version": m["version"], "status": m["status"],
            "n_clips": trained.get("n_clips"), "n_windows": trained.get("n_windows"),
            "mean_abs_end_err": after.get("mean_abs_end_err"),
            "mean_abs_start_err": after.get("mean_abs_start_err"),
            "created_at": m["created_at"],
        })
    return {"kind": kind, "curve": curve}


@router.post("/models/{version}/promote")
def promote_model(version: int):
    if not model_registry.promote(version):
        raise HTTPException(404, "model version not found")
    return {"promoted": version}


@router.post("/models/{version}/reject")
def reject_model(version: int):
    model_registry.reject(version)
    return {"rejected": version}
