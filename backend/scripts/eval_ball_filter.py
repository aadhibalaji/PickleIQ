"""CLI for the ball-filter harness. The evaluation itself lives in
app/analysis/ball_eval.py so the API can run it in-process (job kind
'ball_eval') without shelling out to this script; this file is the terminal
front-end: argparse, the human-readable per-clip table, and the pooled summary.

Compares the ball pipeline WITHOUT the Phase-1 filter (baseline = permissive
detector + trajectory selector) against the pipeline WITH it, on the frames
inside HUMAN rally labels. See ball_eval.py for what each metric means.

Usage (from backend/):
  .venv/Scripts/python.exe scripts/eval_ball_filter.py [--video ID ...]
      [--json PATH] [--params-version N] [--bounce-frame-tol 5]
      [--bounce-dist-m 1.0] [--pos-frame-tol 1] [--diagnose]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.analysis.ball_eval import (DEFAULT_TRACKNET_WEIGHTS,  # noqa: E402
                                    EvalOptions, EvalSkipped, eval_video,
                                    json_safe, pick_videos, resolve_params)


def fmt_clip(m: dict) -> str:
    fr = m["frames"] or 1
    b, fl = m["base"], m["filt"]

    def pct(n, d):
        return f"{100.0 * n / d:5.1f}%" if d else "  n/a "

    def ev(d):
        return ", ".join(f"{k}:{v}" for k, v in sorted(d.items())) or "-"

    lines = [
        f"{m['video']}  ({m['windows']} {m.get('win_source', 'human')} rallies,"
        f" {fr} frames, overlay cover {pct(m['overlay_cov'], fr)},"
        f" {m['raw_cands']} raw cands)",
        f"  det rate      base {pct(b['det'], fr)}   filt {pct(fl['det'], fr)}"
        f" (+interp -> {pct(fl['det'] + fl['interp'], fr)})",
        f"  phys viol     base {pct(b['viol'], b['pairs'])} of {b['pairs']}"
        f"   filt {pct(fl['viol'], fl['pairs'])} of {fl['pairs']}",
        f"  foot-zone FP  base {pct(b['foot'], b['det'])}"
        f"   filt {pct(fl['foot'], fl['det'] + fl['interp'])}",
        f"  max gap       base {max(b['gaps'], default=0)}f"
        f"   filt {max(fl['gaps'], default=0)}f"
        f"   (mean {np.mean(b['gaps']):.1f}f vs {np.mean(fl['gaps']):.1f}f)"
        if b["gaps"] and fl["gaps"] else
        f"  max gap       base {max(b['gaps'], default=0)}f"
        f"   filt {max(fl['gaps'], default=0)}f",
        f"  rejected      vel {fl['rej']['velocity']}, foot {fl['rej']['foot_zone']},"
        f" persist {fl['rej']['persistent']}, arc-outlier {fl['outliers']},"
        f" noise-frag {fl['fragments']}  ({fl['arcs']} arcs)",
        f"  events        base [{ev(b['events'])}]",
        f"                filt [{ev(fl['events'])}]",
    ]
    gt = m.get("gt", {})
    if gt.get("in_windows"):
        def med(v):
            return f"{np.median(v):.1f}" if v else "n/a"

        gtl = [
            f"  GT MARKS      {gt['in_windows']} in rallies ({gt['bounce']} bounce,"
            f" {gt['hit']} hit, {gt['position']} position);"
            f" {gt['disagree']} in/out disagreement(s) vs homography",
            f"  pos err px    base med {med(b['pos_err'])} (n={len(b['pos_err'])},"
            f" miss {b['pos_miss']})   filt med {med(fl['pos_err'])}"
            f" (n={len(fl['pos_err'])}, miss {fl['pos_miss']})",
        ]
        if m.get("tn_recall"):
            tr = m["tn_recall"]
            gtl.append(
                f"  tracknet raw  near {tr['near'][0]}/{tr['near'][1]}"
                f" ({100 * tr['near'][0] / max(1, tr['near'][1]):.0f}%)"
                f"   far {tr['far'][0]}/{tr['far'][1]}"
                f" ({100 * tr['far'][0] / max(1, tr['far'][1]):.0f}%)"
                f"   recall@25px [classical ceiling 56/45]")
        if gt["bounce"]:
            def pr(d):
                p = 100 * d["bounce_matched"] / d["bounce_pred"] if d["bounce_pred"] else 0.0
                r = 100 * d["bounce_matched"] / gt["bounce"]
                return f"P {p:.0f}% R {r:.0f}% ({d['bounce_matched']}/{d['bounce_pred']} pred)"

            def acc(d):
                return (f"{100 * d['inout_ok'] / d['inout_n']:.0f}% of {d['inout_n']}"
                        if d["inout_n"] else "n/a")

            gtl.append(f"  bounce P/R    base {pr(b)}   filt {pr(fl)}   ({gt['bounce']} GT)")
            gtl.append(f"  in/out acc    base {acc(b)}   filt {acc(fl)}")
        lines[1:1] = gtl  # primary metrics directly under the clip header
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--video", action="append", default=[],
                    help="video id(s) to evaluate (default: all human-labeled)")
    ap.add_argument("--json", default="", help="write full metrics JSON here")
    ap.add_argument("--params-version", type=int, default=0,
                    help="learned_params version (default: latest)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="override a param for this run (repeatable)")
    ap.add_argument("--detector", choices=["classical", "tracknet"],
                    default="classical",
                    help="detector for the FILTERED pipeline (baseline stays classical)")
    ap.add_argument("--tracknet-weights", default=str(DEFAULT_TRACKNET_WEIGHTS),
                    help="TrackNet checkpoint for --detector tracknet")
    ap.add_argument("--diagnose", action="store_true",
                    help="attribute every GT disagreement to a pipeline stage, "
                         "slice metrics by condition, save failure galleries")
    ap.add_argument("--tag", default="run",
                    help="config label for gallery paths / JSON (matrix runs)")
    ap.add_argument("--pos-frame-tol", type=int, default=1,
                    help="frames a prediction may be off a mark for position error")
    ap.add_argument("--bounce-frame-tol", type=int, default=5,
                    help="max frame distance for a bounce match")
    ap.add_argument("--bounce-dist-m", type=float, default=1.0,
                    help="max court-plane distance (m) for a bounce match")
    args = ap.parse_args()

    overrides = {}
    for kv in args.set:
        k, _, val = kv.partition("=")
        overrides[k.strip()] = float(val)
        print(f"override: {k.strip()} = {val}")
    _version, params = resolve_params(args.params_version, overrides)

    opts = EvalOptions(
        detector=args.detector, tracknet_weights=args.tracknet_weights,
        diagnose=args.diagnose, tag=args.tag,
        pos_frame_tol=args.pos_frame_tol,
        bounce_frame_tol=args.bounce_frame_tol,
        bounce_dist_m=args.bounce_dist_m, param_overrides=overrides)

    videos = pick_videos(args.video)
    if not videos:
        print("No evaluable videos (need human rally labels + pass2 overlays "
              "+ the clip file on disk).")
        return
    print(f"Evaluating {len(videos)} clip(s): baseline (no filter) vs "
          f"Phase-1 filtering layer\n")
    results = []
    for v in videos:
        try:
            m = eval_video(v, params, opts)
        except EvalSkipped as exc:
            print(f"  !! {v['filename']}: {exc}")
            continue
        results.append(m)
        print(fmt_clip(m))
        if m.get("diag_report"):
            print(m["diag_report"])
        print()

    if results:
        fr = sum(m["frames"] for m in results) or 1
        bd = sum(m["base"]["det"] for m in results)
        fd = sum(m["filt"]["det"] for m in results)
        fi = sum(m["filt"]["interp"] for m in results)
        bv = sum(m["base"]["viol"] for m in results)
        bp = sum(m["base"]["pairs"] for m in results) or 1
        fv = sum(m["filt"]["viol"] for m in results)
        fp = sum(m["filt"]["pairs"] for m in results) or 1
        bf = sum(m["base"]["foot"] for m in results)
        ffz = sum(m["filt"]["foot"] for m in results)
        print("== POOLED ==")
        n_gt = sum(m["gt"]["in_windows"] for m in results)
        if n_gt:
            n_gtb = sum(m["gt"]["bounce"] for m in results)
            for label in ("base", "filt"):
                errs = [e for m in results for e in m[label]["pos_err"]]
                miss = sum(m[label]["pos_miss"] for m in results)
                mt = sum(m[label]["bounce_matched"] for m in results)
                pd_ = sum(m[label]["bounce_pred"] for m in results)
                okn = sum(m[label]["inout_n"] for m in results)
                ok = sum(m[label]["inout_ok"] for m in results)
                print(f"  [{label}] pos err med {np.median(errs):.1f}px"
                      f" (n={len(errs)}, miss {miss})" if errs else
                      f"  [{label}] pos err n/a (miss {miss})", end="")
                if n_gtb:
                    p = 100 * mt / pd_ if pd_ else 0.0
                    print(f" · bounce P {p:.0f}% R {100*mt/n_gtb:.0f}%"
                          + (f" · in/out {100*ok/okn:.0f}% of {okn}" if okn else ""))
                else:
                    print()
            print(f"  ({n_gt} GT marks, {n_gtb} bounces, "
                  f"{sum(m['gt']['disagree'] for m in results)} homography disagreements)")
        print(f"  det rate      base {100*bd/fr:.1f}%   filt {100*fd/fr:.1f}%"
              f" (+interp -> {100*(fd+fi)/fr:.1f}%)")
        print(f"  phys viol     base {100*bv/bp:.1f}%   filt {100*fv/fp:.1f}%")
        print(f"  foot-zone FP  base {100*bf/max(1,bd):.1f}%"
              f"   filt {100*ffz/max(1,fd+fi):.1f}%")

    if args.json:
        Path(args.json).write_text(json.dumps(json_safe(results), indent=2, default=str))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
