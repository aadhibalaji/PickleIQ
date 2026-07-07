"""Human-vs-AI rally comparison: greedy IoU matching, temporal IoU, and
separate start/end boundary errors. Discrepancies are classified as
boundary_drift (rally found, boundaries off) vs missed/extra (whole rally
absent or invented) — they have different fixes and the report routes them
accordingly."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Rally:
    start: int
    end: int
    id: str = ""
    meta: dict = field(default_factory=dict)  # AI rally: Pass 2 boundary metadata


def temporal_iou(a: Rally, b: Rally) -> float:
    inter = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union > 0 else 0.0


def compare(human: list[Rally], ai: list[Rally], fps: float,
            iou_match_threshold: float = 0.1) -> dict:
    """Greedy best-IoU matching (rallies rarely overlap, so greedy ≈ optimal).
    Returns discrepancy dicts plus summary metrics."""
    pairs = sorted(
        ((temporal_iou(h, a), hi, ai_i)
         for hi, h in enumerate(human) for ai_i, a in enumerate(ai)),
        key=lambda t: -t[0],
    )
    matched_h: dict[int, int] = {}
    matched_a: dict[int, int] = {}
    for iou, hi, aii in pairs:
        if iou < iou_match_threshold:
            break
        if hi in matched_h or aii in matched_a:
            continue
        matched_h[hi] = aii
        matched_a[aii] = hi

    discrepancies: list[dict] = []
    ious, start_errs, end_errs = [], [], []
    for hi, h in enumerate(human):
        if hi in matched_h:
            a = ai[matched_h[hi]]
            iou = temporal_iou(h, a)
            se = a.start - h.start  # positive = AI late
            ee = a.end - h.end
            ious.append(iou)
            start_errs.append(se)
            end_errs.append(ee)
            # Where did the AI boundary come from? A boundary derived from a
            # real detected serve / confirmed terminal event is a trustworthy
            # learning signal; one that fell back to the coarse-gate window
            # edge is not (its error is an artefact of padding, not drift).
            m = a.meta or {}
            start_source = m.get("start_source")
            end_source = m.get("end_source")
            confirmed = bool(m.get("confirmed"))
            discrepancies.append({
                "type": "boundary_drift",
                "human_rally": {"start_frame": h.start, "end_frame": h.end},
                "ai_rally": {"start_frame": a.start, "end_frame": a.end},
                "temporal_iou": round(iou, 3),
                "start_error": se,
                "end_error": ee,
                "start_error_ms": round(se / fps * 1000),
                "end_error_ms": round(ee / fps * 1000),
                "start_source": start_source,
                "end_source": end_source,
                "confirmed": confirmed,
                # structurally clean = boundary came from a real detection, not
                # the coarse-gate fallback. The learner adds a magnitude cap.
                "clean_start": start_source == "serve",
                "clean_end": end_source == "terminal_event" and confirmed,
            })
        else:
            discrepancies.append({
                "type": "missed",
                "human_rally": {"start_frame": h.start, "end_frame": h.end},
                "ai_rally": None,
                "temporal_iou": 0.0,
                "start_error": None, "end_error": None,
            })
    for aii, a in enumerate(ai):
        if aii not in matched_a:
            discrepancies.append({
                "type": "extra",
                "human_rally": None,
                "ai_rally": {"start_frame": a.start, "end_frame": a.end},
                "temporal_iou": 0.0,
                "start_error": None, "end_error": None,
            })

    n_matched = len(matched_h)
    abs_start = [abs(x) for x in start_errs]
    abs_end = [abs(x) for x in end_errs]
    summary = {
        "human_rallies": len(human),
        "ai_rallies": len(ai),
        "matched": n_matched,
        "missed": len(human) - n_matched,
        "extra": len(ai) - n_matched,
        "match_rate": round(n_matched / len(human), 3) if human else None,
        "mean_iou": round(sum(ious) / len(ious), 3) if ious else None,
        "mean_start_error_frames": round(sum(start_errs) / len(start_errs), 1) if start_errs else None,
        "mean_end_error_frames": round(sum(end_errs) / len(end_errs), 1) if end_errs else None,
        # Absolute error is the honest "how far off" metric for the scoreboard:
        # signed means can cancel out and hide large two-sided drift.
        "mean_abs_start_error_frames": round(sum(abs_start) / len(abs_start), 1) if abs_start else None,
        "mean_abs_end_error_frames": round(sum(abs_end) / len(abs_end), 1) if abs_end else None,
        "median_start_error_frames": sorted(start_errs)[len(start_errs) // 2] if start_errs else None,
        "median_end_error_frames": sorted(end_errs)[len(end_errs) // 2] if end_errs else None,
        "fps": fps,
    }
    return {"discrepancies": discrepancies, "summary": summary}
