"""Import a self-contained manual-tracking export (schema_version 1) — video
metadata + calibration + ball marks in ONE file — into this machine's DB.

Unlike the legacy ball_gt.json (marks only, keyed by a foreign machine's video
id), this format carries everything needed to stand the clip up:

  * video block  -> a videos row (forged if the clip isn't uploaded yet)
  * calibration  -> a CLIP-LOCAL calibration linked via videos.calibration_id
                    (never touches the global calibration / other clips)
  * ball_marks   -> ball_corrections rows, deduped by (frame, x, y, type)

The video FILE bytes are the only thing not in the export. If the clip is
already uploaded (a videos row with the same filename exists), its marks attach
to that row. Otherwise a row is forged pointing at the expected local path and
you drop the file there.

Usage (from backend/):
  .venv/Scripts/python.exe scripts/import_manual_tracking.py <manual_tracking.json>
  .venv/Scripts/python.exe scripts/import_manual_tracking.py <file> --replace
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import db  # noqa: E402
from app.config import DATA_DIR  # noqa: E402

_MARK_COLS = ("id", "video_id", "frame", "x", "y", "scope", "created_at",
              "event_type", "court_x", "court_y", "in_out", "in_out_suggested",
              "disagree")


def _mark_key(m: dict) -> tuple:
    return (int(m["frame"]), round(float(m["x"])), round(float(m["y"])),
            m.get("event_type") or "position")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("file", type=Path, help="manual_tracking.json export")
    ap.add_argument("--replace", action="store_true",
                    help="clear the clip's existing marks + calibration first")
    args = ap.parse_args()
    if not args.file.exists():
        sys.exit(f"no such file: {args.file}")

    data = json.loads(args.file.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        print(f"warning: schema_version {data.get('schema_version')} != 1 "
              "— attempting anyway")
    v = data["video"]
    cal = data.get("calibration")
    marks = data.get("ball_marks", [])
    filename = v["filename"]

    with db.connect() as con:
        # 1) resolve the local video row: prefer an already-uploaded clip of the
        #    same filename; else forge one keyed by the export's own id, pointing
        #    at the path the file is expected to live at.
        row = con.execute("SELECT id, path FROM videos WHERE filename=?",
                           (filename,)).fetchone()
        forged = False
        if row:
            local_id, local_path = row["id"], row["path"]
        else:
            local_id = v["id"]
            local_path = str(DATA_DIR / "videos" / local_id / filename)
            con.execute(
                "INSERT INTO videos (id, filename, path, fps, width, height,"
                " frame_count, uploaded_at) VALUES (?,?,?,?,?,?,?,?)",
                (local_id, filename, local_path, v.get("fps"), v.get("width"),
                 v.get("height"), v.get("frame_count"), db.now()))
            forged = True

        if args.replace:
            con.execute("DELETE FROM ball_corrections WHERE video_id=?", (local_id,))
            con.execute("DELETE FROM calibrations WHERE video_id=?", (local_id,))

        # 2) clip-local calibration, linked directly on the video row (so it is
        #    used ONLY for this clip and never shadows the global one).
        if cal:
            cal_id = db.new_id()
            con.execute(
                "INSERT INTO calibrations (id, scope, video_id, corner_points,"
                " homography, focal_px, created_at) VALUES (?,?,?,?,?,?,?)",
                (cal_id, "clip", local_id, db.dumps(cal.get("corner_points")),
                 db.dumps(cal["homography"]), cal.get("focal_px"), db.now()))
            con.execute("UPDATE videos SET calibration_id=? WHERE id=?",
                        (cal_id, local_id))

        # 3) marks
        existing = {
            _mark_key({"frame": r["frame"], "x": r["x"], "y": r["y"],
                       "event_type": r["event_type"]})
            for r in con.execute(
                "SELECT frame, x, y, event_type FROM ball_corrections"
                " WHERE video_id=?", (local_id,))}
        ins = skip = 0
        for m in marks:
            key = _mark_key(m)
            if key in existing:
                skip += 1
                continue
            existing.add(key)
            con.execute(
                f"INSERT INTO ball_corrections ({','.join(_MARK_COLS)})"
                f" VALUES ({','.join('?' * len(_MARK_COLS))})",
                (db.new_id(), local_id, int(m["frame"]), float(m["x"]),
                 float(m["y"]), m.get("scope") or "clip_local", db.now(),
                 m.get("event_type") or "position", m.get("court_x"),
                 m.get("court_y"), m.get("in_out"), m.get("in_out_suggested"),
                 int(m.get("disagree") or 0)))
            ins += 1

    print(f"video row:   {'FORGED' if forged else 'existing'}  id={local_id}  "
          f"({v.get('fps')} fps, {v.get('frame_count')} frames)")
    print(f"calibration: {'linked (clip-local)' if cal else 'none in file'}")
    print(f"marks:       inserted {ins}, skipped {skip} already-present")
    file_ok = Path(local_path).exists()
    print(f"video file:  {'PRESENT' if file_ok else 'MISSING'} at {local_path}")
    if not file_ok:
        print("\n  -> drop the actual 'Second 3 mins.mp4' at the path above,"
              " then run Pass 2 (reprocess) so the harness has overlays.")
    else:
        print("\n  -> run Pass 2 on this clip, then eval_ball_filter can score it.")


if __name__ == "__main__":
    main()
