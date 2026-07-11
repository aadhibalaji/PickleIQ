"""Offline precision proxy for audio x visual fusion (Feature 2).

Reads the stored DB state AFTER a Pass 2 run (no re-inference) and reports, per
clip:
  audio_all     total audio onsets extracted for the clip
  audio_inwin   onsets falling inside a Pass 1 rally window (the real candidates)
  fused         Pass 2 overlay evt events carrying an audio-corroboration flag
  serves        evt events promoted to kind='serve'
  demoted       in-window onsets NOT represented by any fused evt — cross-court
                bleed / no visual corroboration. This is the cheap precision
                proxy: a healthy fusion demotes the bleed and keeps real hits.

`demoted = max(0, audio_inwin - fused)` is approximate (one onset can sit near
several events), but needs no manual ground truth. Regression anchors for
"Second 3 mins.mp4": onsets near ~1.87/3.41/158s should end up fused; ~6.11s
(adjacent-court bleed) should land in `demoted`.

Usage (from backend/):
  .venv/Scripts/python.exe scripts/eval_audio_fusion.py [--video ID ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app import db                     # noqa: E402
from app.cv import audio_events, overlays  # noqa: E402


def eval_video(video_id: str) -> dict:
    with db.connect() as con:
        vrow = con.execute("SELECT filename FROM videos WHERE id=?", (video_id,)).fetchone()
        windows = [(r["start_frame"], r["end_frame"]) for r in con.execute(
            "SELECT start_frame, end_frame FROM rally_labels"
            " WHERE video_id=? AND source='pass1' ORDER BY start_frame", (video_id,))]
    onsets = audio_events.load_events(video_id)
    in_win = sum(1 for o in onsets
                 if any(ws <= o["frame_index"] <= we for ws, we in windows))

    fused = serves = evt_total = 0
    for e in overlays.read_range(video_id, "pass2", 0, 10**9):
        ev = e.get("evt")
        if not ev:
            continue
        evt_total += 1
        if ev.get("kind") == "serve":
            serves += 1
        if ev.get("audio"):
            fused += 1
    return {
        "video": video_id, "name": (vrow["filename"] if vrow else "?")[:22],
        "audio_all": len(onsets), "audio_inwin": in_win,
        "evt_total": evt_total, "fused": fused, "serves": serves,
        "demoted": max(0, in_win - fused),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="append", default=[])
    args = ap.parse_args()
    videos = args.video
    if not videos:
        with db.connect() as con:
            videos = [r["video_id"] for r in con.execute(
                "SELECT DISTINCT video_id FROM audio_events")]
    if not videos:
        print("No videos with audio_events. Run Pass 2 (or the audio job) first.")
        return

    print(f"{'clip':22} {'audio':>6} {'inwin':>6} {'evt':>5} "
          f"{'fused':>6} {'serves':>7} {'demoted':>8}")
    for vid in videos:
        r = eval_video(vid)
        print(f"{r['name']:22} {r['audio_all']:6d} {r['audio_inwin']:6d} "
              f"{r['evt_total']:5d} {r['fused']:6d} {r['serves']:7d} {r['demoted']:8d}")


if __name__ == "__main__":
    main()
