"""Versioned, GLOBAL model registry — the weight-based analogue of LearnedParams.

Append-only: training registers a new 'shadow' version; a version is promoted to
'live' only after held-out validation; the live pipeline loads the latest 'live'
version per kind before Pass 2, so a promotion compounds across every future clip
(exactly like learned_params). Rollback = promote a prior version. Weights live
under data/models/<kind>/v<version>.pt.
"""
from __future__ import annotations

from pathlib import Path

from .. import db
from ..config import DATA_DIR

KINDS = ("rally_end", "serve_start", "tracknet")


def model_dir(kind: str) -> Path:
    d = DATA_DIR / "models" / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def register(kind: str, metrics: dict, trained_on: dict, parent: int | None = None,
             origin: str = "manual_cli", status: str = "shadow") -> tuple[int, Path]:
    """Insert a new version row and return (version, path to save the weights to)."""
    with db.connect() as con:
        cur = con.execute(
            "INSERT INTO model_registry (kind, path, status, metrics, parent_version,"
            " trained_on, origin, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (kind, "", status, db.dumps(metrics), parent, db.dumps(trained_on),
             origin, db.now()))
        version = int(cur.lastrowid)
        path = model_dir(kind) / f"v{version}.pt"
        con.execute("UPDATE model_registry SET path=? WHERE version=?", (str(path), version))
    return version, path


def promote(version: int) -> bool:
    with db.connect() as con:
        row = con.execute("SELECT kind FROM model_registry WHERE version=?", (version,)).fetchone()
        if row is None:
            return False
        con.execute("UPDATE model_registry SET status='archived'"
                    " WHERE kind=? AND status='live'", (row["kind"],))
        con.execute("UPDATE model_registry SET status='live', promoted_at=?"
                    " WHERE version=?", (db.now(), version))
    return True


def reject(version: int) -> None:
    with db.connect() as con:
        con.execute("UPDATE model_registry SET status='rejected' WHERE version=?", (version,))


def set_metrics(version: int, metrics: dict) -> None:
    with db.connect() as con:
        con.execute("UPDATE model_registry SET metrics=? WHERE version=?",
                    (db.dumps(metrics), version))


# Validation override: force a specific (usually still-shadow) model into the
# pipeline so the harness can score it without promoting it. Not persisted.
_override: dict = {}


def set_override(kind: str, model) -> None:
    _override[kind] = model


def clear_override() -> None:
    _override.clear()


def latest_live(kind: str) -> tuple[int, str] | None:
    with db.connect() as con:
        row = con.execute(
            "SELECT version, path FROM model_registry WHERE kind=? AND status='live'"
            " ORDER BY version DESC LIMIT 1", (kind,)).fetchone()
    return (row["version"], row["path"]) if row else None


def _build(kind: str):
    if kind == "rally_end":
        from ..cv.bounce import build_bounce_model
        return build_bounce_model()
    # serve_start / tracknet builders are added when those models ship.
    return None


_cache: dict[str, tuple[str, float, object]] = {}  # kind -> (path, mtime, model)


def load_live(kind: str):
    """Return the current live model for `kind` (cached by path+mtime), or None
    if there is no live version, its weights are missing, or the kind has no
    in-process builder yet. Callers fall back to the heuristic on None."""
    if kind in _override:          # validation harness forcing a candidate model
        return _override[kind]
    info = latest_live(kind)
    if info is None:
        return None
    p = Path(info[1])
    if not p.exists():
        return None
    mtime = p.stat().st_mtime
    cached = _cache.get(kind)
    if cached and cached[0] == str(p) and cached[1] == mtime:
        return cached[2]
    model = _build(kind)
    if model is None:
        return None
    import torch
    model.load_state_dict(torch.load(p, map_location="cpu"))
    model.eval()
    _cache[kind] = (str(p), mtime, model)
    return model


def load_live_bundle() -> dict:
    """All live start/end models for Pass 2, loaded once per run."""
    return {"rally_end": load_live("rally_end"), "serve_start": load_live("serve_start")}


def history(kind: str | None = None) -> list[dict]:
    with db.connect() as con:
        if kind:
            rows = con.execute("SELECT * FROM model_registry WHERE kind=?"
                               " ORDER BY version DESC", (kind,)).fetchall()
        else:
            rows = con.execute("SELECT * FROM model_registry ORDER BY version DESC").fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["metrics"] = db.loads(d["metrics"], {})
        d["trained_on"] = db.loads(d["trained_on"], {})
        out.append(d)
    return out
