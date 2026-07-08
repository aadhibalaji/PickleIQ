"""Held-out evaluation matrix for the ball pipeline.

Runs eval_ball_filter.py over the GT-marked clips for every config in the
decision matrix — classical vs TrackNet detector, HSV gate off/on (classical
only; measured from clip-1 marks, so it is the clip-specific candidate), and
tuned arc params vs pre-tuning defaults — then prints a comparison table of
the decision metrics per clip and pooled.

Usage (from backend/):
  .venv/Scripts/python.exe scripts/eval_matrix.py [--video ID ...] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

BACKEND = Path(__file__).resolve().parents[1]
# Run child evals with the SAME interpreter running this script (the old
# hardcoded .venv/Scripts/python.exe was Windows-only and broke elsewhere).
PY = Path(sys.executable)

# Marked clips (Second 3 mins = tuning clip, clip #1 re-upload = held-out).
DEFAULT_VIDEOS = ["760f1e0e95d048fc9ec20ce4c61ccd35",
                  "17be6eb599a146159787a0976c3c54c0"]

PRE_TUNING = ["ball_arc_min_pts=4", "ball_arc_tol_px=6", "ball_arc_gap_frames=10",
              "ball_arc_join_gap_frames=8", "ball_arc_claim_tol_mult=2.5"]
HSV_ON = ["ball_val_min=110", "ball_hue_lo=5", "ball_hue_hi=116"]

CONFIGS = [
    # (tag, detector, extra --set overrides)
    ("c-tuned", "classical", []),
    ("c-tuned-hsv", "classical", HSV_ON),
    ("c-pre", "classical", PRE_TUNING),
    ("c-pre-hsv", "classical", PRE_TUNING + HSV_ON),
    ("t-tuned", "tracknet", []),
    ("t-pre", "tracknet", PRE_TUNING),
]


def run_config(tag, detector, sets, videos, out_dir):
    out_json = out_dir / f"{tag}.json"
    cmd = [str(PY), "-u", str(BACKEND / "scripts" / "eval_ball_filter.py"),
           "--diagnose", "--tag", tag, "--detector", detector,
           "--json", str(out_json)]
    for vid in videos:
        cmd += ["--video", vid]
    for s in sets:
        cmd += ["--set", s]
    log = out_dir / f"{tag}.log"
    print(f"== {tag} ({detector}{' +' + ','.join(sets) if sets else ''}) -> {log.name}")
    with open(log, "w", encoding="utf-8") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                           cwd=str(BACKEND), encoding="utf-8",
                           env={"PYTHONIOENCODING": "utf-8",
                                **__import__("os").environ})
    if r.returncode != 0 or not out_json.exists():
        print(f"   !! {tag} failed (exit {r.returncode}) — see {log}")
        return None
    return json.loads(out_json.read_text())


def summarize(tag, results):
    rows = []
    for m in results:
        fl = m["filt"]
        errs = fl["pos_err"]
        n_marks = m["gt"]["in_windows"]
        onball = sum(1 for e in errs if e <= 25.0)
        rows.append({
            "clip": m["video"][:18], "tag": tag,
            "med_px": float(np.median(errs)) if errs else float("nan"),
            "onball_pct": 100.0 * onball / max(1, n_marks),
            "b_matched": fl["bounce_matched"], "b_pred": fl["bounce_pred"],
            "b_gt": m["gt"]["bounce"],
            "inout": (100.0 * fl["inout_ok"] / fl["inout_n"]) if fl["inout_n"] else float("nan"),
            "foot_pct": 100.0 * fl["foot"] / max(1, fl["det"] + fl["interp"]),
            "fp_by_rally": (m.get("diag") or {}).get("fp_bounces_by_rally", {}),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="append", default=[])
    ap.add_argument("--out", default=str(BACKEND.parent / "data" / "eval_diag" / "matrix"))
    args = ap.parse_args()
    videos = args.video or DEFAULT_VIDEOS
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for tag, det, sets in CONFIGS:
        res = run_config(tag, det, sets, videos, out_dir)
        if res:
            all_rows += summarize(tag, res)

    print("\n== MATRIX (filtered pipeline; on-ball = final output <=25px of mark) ==")
    print(f"{'config':13} {'clip':18} {'med px':>7} {'on-ball':>8} "
          f"{'bounce R':>9} {'bounce P':>9} {'in/out':>7} {'footFP':>7}")
    for r in all_rows:
        br = f"{r['b_matched']}/{r['b_gt']}"
        bp = f"{r['b_matched']}/{r['b_pred']}"
        io = f"{r['inout']:.0f}%" if not math.isnan(r["inout"]) else "n/a"
        print(f"{r['tag']:13} {r['clip']:18} {r['med_px']:7.1f} "
              f"{r['onball_pct']:7.1f}% {br:>9} {bp:>9} {io:>7} {r['foot_pct']:6.1f}%")
    print("\nunmatched predicted bounces per rally (precision inputs for the "
          "exhaustively-marked windows):")
    for r in all_rows:
        if r["fp_by_rally"]:
            print(f"  {r['tag']:13} {r['clip']:18} "
                  + " ".join(f"{k}:{n}" for k, n in sorted(r["fp_by_rally"].items())))
    print(f"\nper-config logs/JSON/galleries under {out_dir}")


if __name__ == "__main__":
    main()
