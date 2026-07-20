"""Export all manual (human-entered) tracking data to JSON, one bundle per clip,
with the clip's video copied alongside its JSON.

Manual data = ball ground-truth marks (ball_corrections), human rally/serve
labels (rally_labels source='human'), and user notes. Anything AI-derived
(pass1/pass2 labels, overlays, refiner output) is excluded.

Output layout:
    exports/manual_tracking/
        <safe-clipname>__<shortid>/
            manual_tracking.json
            <original video filename>

Run:  python backend/scripts/export_manual_tracking.py
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

# Make `app` importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import db  # noqa: E402
from app.config import REPO_ROOT  # noqa: E402

OUT_ROOT = REPO_ROOT / "exports" / "manual_tracking"
SCHEMA_VERSION = 1


def _safe_name(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("_") or "clip"


def _clip_calibration(con, video_id: str) -> dict | None:
    """The calibration in effect for this clip: its clip-local override if set,
    else the most recent global one."""
    v = con.execute("SELECT calibration_id FROM videos WHERE id=?", (video_id,)).fetchone()
    cal = None
    if v and v["calibration_id"]:
        cal = con.execute("SELECT * FROM calibrations WHERE id=?",
                          (v["calibration_id"],)).fetchone()
    if cal is None:
        cal = con.execute(
            "SELECT * FROM calibrations WHERE scope='global' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    if cal is None:
        return None
    return {
        "id": cal["id"], "scope": cal["scope"],
        "corner_points": db.loads(cal["corner_points"]),
        "homography": db.loads(cal["homography"]),
        "focal_px": cal["focal_px"],
    }


def _ball_marks(con, video_id: str) -> list[dict]:
    rows = con.execute(
        "SELECT * FROM ball_corrections WHERE video_id=? ORDER BY frame", (video_id,)
    ).fetchall()
    out = db.rows_to_dicts(rows)
    for r in out:  # legacy NULL event_type == plain position mark
        if not r.get("event_type"):
            r["event_type"] = "position"
    return out


def _human_labels(con, video_id: str) -> list[dict]:
    rows = con.execute(
        "SELECT * FROM rally_labels WHERE video_id=? AND source='human' ORDER BY start_frame",
        (video_id,),
    ).fetchall()
    return [{"id": r["id"], "start_frame": r["start_frame"],
             "end_frame": r["end_frame"], "meta": db.loads(r["meta"], {})}
            for r in rows]


def _notes(con, video_id: str) -> list[dict]:
    rows = con.execute(
        "SELECT * FROM user_notes WHERE video_id=? ORDER BY created_at", (video_id,)
    ).fetchall()
    return db.rows_to_dicts(rows)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = []
    with db.connect() as con:
        videos = con.execute("SELECT * FROM videos ORDER BY uploaded_at").fetchall()
        for v in videos:
            vid = v["id"]
            ball = _ball_marks(con, vid)
            labels = _human_labels(con, vid)
            notes = _notes(con, vid)
            if not (ball or labels or notes):
                continue  # no manual data for this clip

            safe = f"{_safe_name(v['filename'])}__{vid[:8]}"
            clip_dir = OUT_ROOT / safe
            clip_dir.mkdir(parents=True, exist_ok=True)

            # Copy the video next to the JSON (self-contained bundle).
            src = Path(v["path"])
            video_out_name = None
            copied = False
            if src.exists():
                video_out_name = src.name
                dst = clip_dir / video_out_name
                if not (dst.exists() and dst.stat().st_size == src.stat().st_size):
                    shutil.copy2(src, dst)
                copied = True

            bounces = sum(1 for b in ball if b.get("event_type") == "bounce")
            hits = sum(1 for b in ball if b.get("event_type") == "hit")
            positions = sum(1 for b in ball if b.get("event_type") == "position")

            bundle = {
                "schema_version": SCHEMA_VERSION,
                "video": {
                    "id": vid,
                    "filename": v["filename"],
                    "video_file": video_out_name,  # relative to this JSON; null if source missing
                    "source_path": v["path"],
                    "fps": v["fps"], "width": v["width"], "height": v["height"],
                    "frame_count": v["frame_count"],
                },
                "calibration": _clip_calibration(con, vid),
                "counts": {
                    "ball_marks": len(ball),
                    "ball_bounces": bounces, "ball_hits": hits, "ball_positions": positions,
                    "human_rally_labels": len(labels),
                    "notes": len(notes),
                },
                "ball_marks": ball,
                "human_rally_labels": labels,
                "notes": notes,
            }
            (clip_dir / "manual_tracking.json").write_text(
                json.dumps(bundle, indent=2), encoding="utf-8")

            manifest.append({
                "video_id": vid, "filename": v["filename"], "dir": safe,
                "video_copied": copied, **bundle["counts"],
            })
            status = "video copied" if copied else "VIDEO MISSING at source"
            print(f"  {v['filename']:<24} {vid[:8]}  "
                  f"ball={len(ball):>3} (b{bounces}/h{hits}/p{positions})  "
                  f"labels={len(labels):>2}  notes={len(notes):>2}  [{status}]")

    (OUT_ROOT / "manifest.json").write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "clips": manifest}, indent=2),
        encoding="utf-8")
    print(f"\nExported {len(manifest)} clip bundle(s) to {OUT_ROOT}")


if __name__ == "__main__":
    main()
