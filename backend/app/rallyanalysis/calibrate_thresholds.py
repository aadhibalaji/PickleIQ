"""Calibration tool for FORM_METRIC_THRESHOLDS / POSITIONING_METRIC_THRESHOLDS.

TWO MODES, and the difference matters more than anything else in this file:

  SUPERVISED (default whenever db.coaching_marks holds labels for the focus)
    Sweeps each metric's threshold and picks the value that best agrees with
    an EXPERT'S OWN verdicts on real moments — maximizing F1 as scored by
    label_eval, the identical rule backend/scripts/eval_coaching.py reports.
    This is the only mode that can tell you a threshold is firing on the
    WRONG moments rather than merely on unusual ones.

  PERCENTILE (fallback when nothing is labeled yet)
    The original behaviour, kept because a brand-new install has no labels.
    Be aware of what it actually does: setting a threshold at the p90 of its
    own observed deviation distribution flags ~10% of frames BY
    CONSTRUCTION, whether or not anything is wrong. It is a starting point
    for collecting labels, never a validated answer. Prefer labeling ~100+
    moments and re-running supervised.

Percentile-mode detail: percentiles are computed two ways per metric, since a
metric almost always inside its band (deviation 0) makes the "over ALL
frames" percentile misleading at the low end (p90 of a mostly-zero population
can itself be 0, which is not a usable threshold).

Percentiles are computed two ways per metric, since a metric that's almost
always inside its acceptable band (deviation 0) makes the "over ALL frames"
percentile misleading at the low end (p90 of a mostly-zero population can
itself be 0, which is not a usable threshold):
  - over ALL frames (including in-band 0s) — answers "what value would only
    the worst ~X% of every frame this metric ever saw cross?"
  - over EXCURSION frames only (deviation != 0, i.e. already outside the
    acceptable band at all) — answers "within the moments that are already
    somewhat off, what value marks the worst ~X% of THOSE?"
Both are printed; the recommendation falls back to the excursion-only view
when the all-frames p90 is 0 (too rare an excursion for the full-population
percentile to mean anything).

Read-only: never writes to frame_metrics.py's threshold constants itself —
run it, look at the numbers, then edit FORM_METRIC_THRESHOLDS /
POSITIONING_METRIC_THRESHOLDS by hand using the printed diff-ready dict.

Usage: python -m app.rallyanalysis.calibrate_thresholds [form|positioning]
                                                        [--percentile]
"""
from __future__ import annotations

import sys

from .. import db
from . import coaching, frame_metrics, label_eval

PERCENTILES = (50, 75, 90, 95, 99)

# Threshold candidates for the supervised sweep, as multiples of each
# metric's own current shipped value. Spanning roughly a third to triple
# covers "we are far too twitchy" through "we are missing almost everything"
# without pretending to more resolution than ~100 labeled moments support —
# a finer grid would just overfit the label set.
SWEEP_FACTORS = (0.33, 0.5, 0.67, 0.85, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)


def done_video_ids() -> list[str]:
    with db.connect() as con:
        rows = con.execute(
            "SELECT DISTINCT video_id FROM jobs WHERE kind='pass2' AND status='done'").fetchall()
    return [r["video_id"] for r in rows]


def collect_deviations(video_ids: list[str], focus: str) -> dict[str, list[float]]:
    """{metric: [|deviation|, ...]} across every real reading, every player,
    every given video — the same continuous series spike detection scores
    against, pooled raw instead of grouped into spikes. One bad/unready clip
    is skipped (logged), never aborts the whole run."""
    by_metric: dict[str, list[float]] = {m: [] for m in frame_metrics.targets_and_category(focus)[0]}
    for vid in video_ids:
        try:
            result = coaching._build_vectors_meta(vid, focus=focus)
        except Exception as exc:  # noqa: BLE001 — calibration must survive one bad clip
            print(f"  skipping {vid}: {exc}", file=sys.stderr)
            continue
        series_by_player = result[8]
        for series in series_by_player.values():
            for fs in series:
                for m, d in fs["devs"].items():
                    if d is not None:
                        by_metric[m].append(abs(d))
    return by_metric


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def report(focus: str, video_ids: list[str]) -> dict[str, float]:
    by_metric = collect_deviations(video_ids, focus)
    thresholds = frame_metrics.metric_thresholds(focus)
    print(f"\n=== {focus} ===")
    recommended: dict[str, float] = {}
    for metric, values in by_metric.items():
        if not values:
            print(f"{metric}: no data")
            continue
        nonzero = [v for v in values if v > 0]
        excursion_rate = len(nonzero) / len(values)
        all_pcts = {p: percentile(values, p) for p in PERCENTILES}
        exc_pcts = {p: percentile(nonzero, p) for p in PERCENTILES} if nonzero else {}
        current = thresholds[metric]
        flagged_now = sum(1 for v in values if v >= current)

        print(f"{metric}: n={len(values)} frames, excursion_rate={excursion_rate:.1%} "
              f"(current threshold {current} flags {flagged_now}/{len(values)} = {flagged_now/len(values):.1%})")
        print("  all-frames pctl:  " + " ".join(f"p{p}={all_pcts[p]:.2f}" for p in PERCENTILES))
        if nonzero:
            print("  excursion pctl:   " + " ".join(f"p{p}={exc_pcts[p]:.2f}" for p in PERCENTILES))

        rec = all_pcts[90]
        if rec <= 0 and nonzero:
            rec = exc_pcts[50]  # excursions too rare for the full-population p90 to mean anything
        recommended[metric] = round(max(rec, 0.1), 2)
    return recommended


def supervised_report(focus: str) -> dict[str, float] | None:
    """Coordinate-ascent sweep: tune one metric at a time, holding the rest
    at their current values, keeping any improvement. Returns None when
    nothing is labeled for this focus (caller falls back to percentiles).

    Coordinate ascent rather than a joint grid because the metrics interact
    only weakly (each spike has ONE dominant metric) while a joint sweep is
    exponential in metric count — and because a per-metric trace is
    something a human can actually read and sanity-check.

    Recall is shared across metrics (any metric can find a given bad span),
    so F1 here is always the WHOLE focus's F1 at that candidate value, not
    the metric's own. That is deliberate: it prevents a metric being tuned to
    look good in isolation while degrading the report as a whole.
    """
    video_ids = label_eval.labeled_video_ids(focus)
    if not video_ids:
        return None

    # Load each labeled clip's series and marks ONCE, then sweep in memory.
    loaded = []
    for vid in video_ids:
        marks = label_eval.load_marks(vid, focus)
        if not marks:
            continue
        try:
            series, fps = label_eval.load_series(vid, focus)
        except Exception as exc:  # noqa: BLE001 — one bad clip must not abort calibration
            print(f"  skipping {vid}: {exc}", file=sys.stderr)
            continue
        loaded.append((series, fps, marks))
    if not loaded:
        return None

    def f1_at(thresholds: dict) -> float | None:
        tp = fp = fn = n_bad = 0
        for series, fps, marks in loaded:
            spikes = label_eval.spikes_from_series(series, fps, focus, thresholds)
            s = label_eval.score(spikes, marks)
            tp, fp, fn, n_bad = tp + s["tp"], fp + s["fp"], fn + s["fn"], n_bad + s["n_bad_spans"]
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = (n_bad - fn) / n_bad if n_bad else None
        if precision is None or recall is None or (precision + recall) == 0:
            return None
        return 2 * precision * recall / (precision + recall)

    current = dict(frame_metrics.metric_thresholds(focus))
    n_marks = sum(len(m) for _s, _f, m in loaded)
    base = f1_at(current)
    print(f"\n=== {focus} (supervised: {len(loaded)} clips, {n_marks} marks) ===")
    print(f"  current thresholds F1 = {base if base is not None else float('nan'):.3f}")
    if n_marks < 100:
        print(f"  WARNING: {n_marks} marks is a thin basis — the sweep will overfit. "
              "Aim for 100+ before trusting these numbers.")

    for metric in list(current):
        best_val, best_f1 = current[metric], base
        trace = []
        for factor in SWEEP_FACTORS:
            candidate = dict(current)
            candidate[metric] = round(max(0.1, current[metric] * factor), 2)
            f1 = f1_at(candidate)
            trace.append((candidate[metric], f1))
            if f1 is not None and (best_f1 is None or f1 > best_f1):
                best_val, best_f1 = candidate[metric], f1
        print(f"  {metric}: {current[metric]} -> {best_val} (F1 "
              f"{base if base is not None else float('nan'):.3f} -> "
              f"{best_f1 if best_f1 is not None else float('nan'):.3f})")
        print("    " + " ".join(f"{v}:{'n/a' if f is None else format(f, '.2f')}" for v, f in trace))
        current[metric] = best_val
        base = best_f1

    print(f"  final F1 = {base if base is not None else float('nan'):.3f}")
    return current


def _print_dict(focus: str, recs: dict[str, float]) -> None:
    const_name = ("POSITIONING_METRIC_THRESHOLDS" if focus == "positioning"
                  else "FORM_METRIC_THRESHOLDS")
    print(f"{const_name} = {{")
    for m, v in recs.items():
        print(f"    {m!r}: {v},")
    print("}")


if __name__ == "__main__":
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    force_percentile = "--percentile" in sys.argv
    focus_arg = argv[0] if argv else None
    focuses = [focus_arg] if focus_arg else ["form", "positioning"]

    all_recs: dict[str, dict[str, float]] = {}
    modes: dict[str, str] = {}
    for focus in focuses:
        recs = None if force_percentile else supervised_report(focus)
        if recs is not None:
            all_recs[focus], modes[focus] = recs, "supervised"
            continue
        if not force_percentile:
            print(f"\n[{focus}] no expert labels — falling back to percentiles. These "
                  "flag a fixed share of frames by construction; label moments in the "
                  "viewer and re-run for a validated answer.")
        video_ids = done_video_ids()
        all_recs[focus], modes[focus] = report(focus, video_ids), "percentile"

    print("\nRecommended thresholds — paste into frame_metrics.py:")
    for focus, recs in all_recs.items():
        print(f"# {focus}: {modes[focus]}"
              + ("" if modes[focus] == "supervised" else " (UNVALIDATED — see module docstring)"))
        _print_dict(focus, recs)
