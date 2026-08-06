"""Scores coaching spike detection against EXPERT GROUND TRUTH — the
regression test for every change to frame_metrics' metrics and thresholds.

Why it exists: coaching thresholds were previously set to the p90 of their
own observed deviation distribution, which flags a fixed ~10% of frames
whether or not anything is actually wrong, and cannot tell you the detector
is firing on the WRONG moments. This harness answers that question against
labels a real player made (db.coaching_marks, written by the viewer's
coaching-marking mode).

Metrics (see app/rallyanalysis/label_eval.py for the exact matching rule):

  precision  of the moments we flag, the share the expert called 'bad'.
             Spikes the expert never reviewed are UNJUDGED and excluded —
             they measure labeling budget, not detector error.
  recall     of the 'bad' spans the expert marked while free-scrubbing, the
             share we found at all. Only freeform marks count: spike-sourced
             ones exist where a spike already fired, so including them would
             make recall trivially 1.0.
  f1         the single number to compare runs by.
  per-metric precision, so a bad number can be attributed to the metric
             actually responsible instead of blamed on the focus as a whole.

Read-only: touches no thresholds, writes nothing. Run it before and after a
change; a change that doesn't move F1 didn't help.

Usage (from backend/):
  .venv/Scripts/python.exe scripts/eval_coaching.py [--focus form|positioning]
      [--video ID ...] [--json PATH] [--show-missed]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rallyanalysis import label_eval  # noqa: E402

FOCUSES = ("form", "positioning")


def _fmt(value: float | None, width: int = 6) -> str:
    """Rates are None when undefined (nothing labeled yet) — print that
    honestly rather than as 0.00, which would read as a real bad score."""
    return f"{value:>{width}.2f}" if value is not None else f"{'n/a':>{width}}"


def evaluate(video_ids: list[str], focus: str) -> list[dict]:
    results = []
    for vid in video_ids:
        marks = label_eval.load_marks(vid, focus)
        if not marks:
            continue
        try:
            spikes = label_eval.spikes_for_video(vid, focus)
        except Exception as exc:  # noqa: BLE001 — one unready clip must not abort the run
            print(f"  skipping {vid}: {exc}", file=sys.stderr)
            continue
        res = label_eval.score(spikes, marks)
        res["video_id"] = vid
        res["by_metric"] = label_eval.score_by_metric(spikes, marks)
        results.append(res)
    return results


def pooled(results: list[dict]) -> dict:
    """One score over every clip's raw counts — NOT the mean of per-clip
    rates, which would weight a 3-mark clip the same as a 60-mark one."""
    tp = sum(r["tp"] for r in results)
    fp = sum(r["fp"] for r in results)
    fn = sum(r["fn"] for r in results)
    n_bad = sum(r["n_bad_spans"] for r in results)
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = (n_bad - fn) / n_bad if n_bad else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and (precision + recall) > 0
          else None)
    return {"tp": tp, "fp": fp, "fn": fn,
            "unjudged": sum(r["unjudged"] for r in results),
            "n_spikes": sum(r["n_spikes"] for r in results),
            "n_marks": sum(r["n_marks"] for r in results), "n_bad_spans": n_bad,
            "precision": precision, "recall": recall, "f1": f1}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--focus", choices=FOCUSES, help="default: both")
    ap.add_argument("--video", action="append", dest="videos",
                    help="repeatable; default: every clip with marks for the focus")
    ap.add_argument("--json", dest="json_path", help="write full results here")
    ap.add_argument("--show-missed", action="store_true",
                    help="list the frame spans the detector failed to find")
    args = ap.parse_args()

    payload: dict = {}
    for focus in ([args.focus] if args.focus else list(FOCUSES)):
        video_ids = args.videos or label_eval.labeled_video_ids(focus)
        results = evaluate(video_ids, focus)

        print(f"\n=== {focus} ===")
        if not results:
            print("  no labeled clips — mark some moments in the viewer first")
            payload[focus] = {"clips": [], "pooled": None}
            continue

        print(f"  {'clip':<14} {'spikes':>6} {'marks':>6} {'prec':>6} {'rec':>6} "
              f"{'f1':>6} {'tp':>4} {'fp':>4} {'fn':>4} {'unjdg':>6}")
        for r in results:
            print(f"  {r['video_id'][:14]:<14} {r['n_spikes']:>6} {r['n_marks']:>6} "
                  f"{_fmt(r['precision'])} {_fmt(r['recall'])} {_fmt(r['f1'])} "
                  f"{r['tp']:>4} {r['fp']:>4} {r['fn']:>4} {r['unjudged']:>6}")

        p = pooled(results)
        print(f"  {'== POOLED ==':<14} {p['n_spikes']:>6} {p['n_marks']:>6} "
              f"{_fmt(p['precision'])} {_fmt(p['recall'])} {_fmt(p['f1'])} "
              f"{p['tp']:>4} {p['fp']:>4} {p['fn']:>4} {p['unjudged']:>6}")

        by_metric: dict[str, dict] = {}
        for r in results:
            for metric, m in r["by_metric"].items():
                agg = by_metric.setdefault(metric, {"n_spikes": 0, "tp": 0, "fp": 0, "unjudged": 0})
                for k in ("n_spikes", "tp", "fp", "unjudged"):
                    agg[k] += m[k]
        if by_metric:
            print("\n  per-metric precision (which metric causes the false positives):")
            for metric, m in sorted(by_metric.items()):
                prec = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) else None
                print(f"    {metric:<32} spikes={m['n_spikes']:>4} tp={m['tp']:>3} "
                      f"fp={m['fp']:>3} unjudged={m['unjudged']:>4} prec={_fmt(prec)}")
                m["precision"] = prec

        if args.show_missed:
            missed = [(r["video_id"], m) for r in results for m in r["missed"]]
            print(f"\n  missed 'bad' spans ({len(missed)}) — what the detector is blind to:")
            for vid, m in missed:
                note = f"  {m['note']}" if m.get("note") else ""
                print(f"    {vid[:14]:<14} player {m['player_id']} "
                      f"frames {m['start_frame']}-{m['end_frame']}"
                      f"{'  metric=' + m['metric'] if m.get('metric') else ''}{note}")

        payload[focus] = {"clips": results, "pooled": p, "by_metric": by_metric}

    if args.json_path:
        Path(args.json_path).write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nwrote {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
