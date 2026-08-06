"""Scoring spike detection against EXPERT LABELS (db.coaching_marks) — the
measurement layer that replaces "tune the numbers and see if the tips read
better" with precision/recall/F1.

Why this module exists: every threshold in frame_metrics was previously
derived from the p90 of its own observed deviation distribution (see
calibrate_thresholds), which flags a fixed ~10% of frames whether or not
anything is actually wrong. A percentile cannot tell you the detector is
firing on the wrong moments. Human labels can.

Two label sources, measuring two different things — both are needed:
  - source='spike'    the expert's verdict on a moment the detector ALREADY
                      found. Joined by spike_id == contact_id. Yields
                      PRECISION (of what we flag, how much is really bad).
  - source='freeform' a span the expert marked while scrubbing, independent
                      of detection. Matched by player + frame overlap. Yields
                      RECALL (of what is really bad, how much do we find) —
                      unobtainable from spike labels, which by construction
                      only exist where the detector already fired.

Shared by backend/scripts/eval_coaching.py (reporting) and
calibrate_thresholds.py (supervised threshold sweep) so both score matches
identically — a sweep optimizing a different rule than the harness reports
would be worse than no sweep at all.
"""
from __future__ import annotations

from .. import db
from . import coaching, frame_metrics

# A freeform span and a spike count as the same moment when they overlap at
# all (and concern the same player). Deliberately permissive: an expert
# dragging a rough span around "this bit here" is not trying to agree with a
# detector's exact frame bounds, and demanding tight overlap would score
# bookkeeping rather than whether the moment was found.
MIN_OVERLAP_FRAMES = 1


def load_marks(video_id: str, focus: str) -> list[dict]:
    with db.connect() as con:
        rows = con.execute(
            "SELECT * FROM coaching_marks WHERE video_id=? AND focus=?"
            " ORDER BY start_frame", (video_id, focus)).fetchall()
    return db.rows_to_dicts(rows)


def labeled_video_ids(focus: str) -> list[str]:
    with db.connect() as con:
        rows = con.execute(
            "SELECT DISTINCT video_id FROM coaching_marks WHERE focus=?", (focus,)).fetchall()
    return [r["video_id"] for r in rows]


def load_series(video_id: str, focus: str) -> tuple[dict, float]:
    """(series_by_player, fps) for one clip — the expensive part of scoring
    (DB reads + overlay decode + per-frame metric computation), pulled out so
    a threshold SWEEP can rebuild spikes hundreds of times from one load.
    Without this, calibration re-reads every overlay per candidate value."""
    result = coaching._build_vectors_meta(video_id, focus=focus)
    return result[8], result[3]


def spikes_from_series(series_by_player: dict, fps: float, focus: str,
                        thresholds: dict) -> list[dict]:
    """Spikes at an arbitrary threshold set, from an already-loaded series.
    contact_ids are synthetic — a spike existing only at a candidate
    threshold has no production identity; scoring matches by overlap anyway."""
    _targets, category_map = frame_metrics.targets_and_category(focus)
    out: list[dict] = []
    for pid, series in series_by_player.items():
        ordered = sorted(series, key=lambda fs: fs["frame"])
        for sp in frame_metrics.detect_spikes(ordered, thresholds, fps, category_map):
            out.append({
                "contact_id": f"sweep_{pid}_{sp['dominant_metric']}_{sp['peak_frame']}",
                "player_id": pid,
                "start_frame": sp["start_frame"], "end_frame": sp["end_frame"],
                "peak_frame": sp["peak_frame"],
                "dominant_metric": sp["dominant_metric"], "category": sp["category"],
            })
    return out


def spikes_for_video(video_id: str, focus: str,
                      thresholds: dict | None = None) -> list[dict]:
    """Every detected spike for one clip as {contact_id, player_id,
    start_frame, end_frame, peak_frame, dominant_metric, category}.

    thresholds=None returns the PRODUCTION spikes (real contact_ids, exactly
    what a report would be built from). Passing an override re-runs
    detect_spikes over the same frame series at different bars — the sweep
    path — where contact_ids are synthetic, since a spike that only exists at
    a hypothetical threshold has no production identity.

    Nothing downstream joins on contact_id (see score()): both paths are
    matched to labels by player + frame overlap, so the two are
    interchangeable for scoring.

    Approximation in the sweep path: the series is every rally concatenated
    per player (what _build_vectors_meta returns) whereas production detects
    per rally. detect_spikes' GAP_BREAK_S guard breaks a run across a rally
    boundary's frame jump anyway, so the two agree except where two rallies
    are frame-adjacent.
    """
    (vectors_meta, focus, _mt, fps, _r, _os, _vp, _gc, series_by_player) = \
        coaching._build_vectors_meta(video_id, focus=focus)

    if thresholds is None:
        out = []
        for vec, contact in vectors_meta:
            span = contact.get("span") or {}
            start = coaching._span_frame(span.get("start_s"), fps)
            end = coaching._span_frame(span.get("end_s"), fps)
            if start is None or end is None:
                continue
            out.append({
                "contact_id": vec.contact_id, "player_id": contact.get("player_id"),
                "start_frame": start, "end_frame": end,
                "peak_frame": contact.get("frame_idx"),
                "dominant_metric": span.get("metric"), "category": vec.category,
            })
        return out

    return spikes_from_series(series_by_player, fps, focus, thresholds)


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return min(a_end, b_end) - max(a_start, b_start) >= MIN_OVERLAP_FRAMES - 1


def score(spikes: list[dict], marks: list[dict]) -> dict:
    """Precision/recall/F1 of spikes against expert marks.

    Everything is matched by player + frame overlap, never by spike_id — so
    the same rule scores production spikes and hypothetical swept ones alike
    (a spike that exists only at a candidate threshold has no id to join on).

    A spike counts as a TRUE POSITIVE when an overlapping mark for that
    player says 'bad', a FALSE POSITIVE when one explicitly says 'fine'/
    'good'. A spike nobody looked at is neither: it is UNJUDGED and excluded
    from precision, because counting unreviewed spikes as wrong would punish
    the detector for the expert's labeling budget rather than its own errors.
    When several marks overlap one spike, 'bad' wins — an expert who marked a
    bad span and a neighbouring fine one has still identified a real problem
    at that moment.

    A 'bad' FREEFORM span with no overlapping spike is a FALSE NEGATIVE — the
    misses. Only freeform marks can contribute here: spike-sourced ones exist
    only where a spike already fired, so counting them would make recall
    trivially 1.0. That is precisely why freeform labeling is required rather
    than an optional extra.
    """
    freeform = [m for m in marks if m["source"] == "freeform"]
    bad_spans = [m for m in freeform if m["verdict"] == "bad"]

    def _verdict_for(sp: dict) -> str | None:
        hits = [m["verdict"] for m in marks
                if m["player_id"] == sp["player_id"] and _overlaps(
                    sp["start_frame"], sp["end_frame"], m["start_frame"], m["end_frame"])]
        if not hits:
            return None
        return "bad" if "bad" in hits else hits[0]

    tp = fp = unjudged = 0
    for sp in spikes:
        verdict = _verdict_for(sp)
        if verdict is None:
            unjudged += 1
        elif verdict == "bad":
            tp += 1
        else:
            fp += 1

    missed = [m for m in bad_spans
              if not any(sp["player_id"] == m["player_id"] and _overlaps(
                  sp["start_frame"], sp["end_frame"], m["start_frame"], m["end_frame"])
                  for sp in spikes)]
    fn = len(missed)

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = (len(bad_spans) - fn) / len(bad_spans) if bad_spans else None
    # `is not None` rather than truthiness: a real 0.0 precision is a
    # measurement, not a missing one, and must not silently become f1=None.
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    return {"n_spikes": len(spikes), "tp": tp, "fp": fp, "fn": fn,
            "unjudged": unjudged, "n_bad_spans": len(bad_spans),
            "n_marks": len(marks), "precision": precision, "recall": recall,
            "f1": f1, "missed": missed}


def score_by_metric(spikes: list[dict], marks: list[dict]) -> dict[str, dict]:
    """Same scoring, split by each spike's dominant metric — shows WHICH
    metric is responsible for the false positives, which a pooled number
    hides. Recall is deliberately absent per metric: a missed span has no
    dominant metric (nothing fired), so it cannot be attributed to one."""
    out: dict[str, dict] = {}
    for metric in {sp["dominant_metric"] for sp in spikes}:
        subset = [sp for sp in spikes if sp["dominant_metric"] == metric]
        s = score(subset, marks)
        out[metric] = {k: s[k] for k in ("n_spikes", "tp", "fp", "unjudged", "precision")}
    return out
