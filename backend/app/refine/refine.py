"""Orchestrator: Pass 1 candidate windows in -> refinement records out.

Per candidate window: load tracking (window ± margin only), Stage A features,
Stage B P(live), Stage C temporal smoothing, Stage D serve gate + boundary
snapping — then emit one record per surviving segment (a window containing
two rallies emits two records with the same ``pass1_window_id``) or a single
rejection record.

Output contract (per record)::

    {"pass1_window_id": str, "pass1_start": s, "pass1_end": s,
     "pass1_start_frame": int, "pass1_end_frame": int,
     "verdict": "rally"|"rejected",
     "rejection_reason": null|"no_serve"|"low_live_ratio"|"too_short",
     "refined_start": s|null, "refined_end": s|null,
     "refined_start_frame": int|null, "refined_end_frame": int|null,
     "confidence": {"mean_p_live": f, "serve_gate_score": f,
                    "net_crossings_per_10s": f},
     "splits": [[s,e], ...]}   # sibling segments when the window split, else []

All timestamps are seconds from clip start; frames are also included since
the rest of the repo speaks frames. JSON-only output for now — writes no
``rally_labels`` rows.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .classifiers import LiveClassifier, make_classifier
from .config import RefineConfig, load_config
from .features import Features, compute_features
from .io_loaders import (Pass1Window, TrackingData, load_pass1_windows,
                         load_tracking, load_video)
from .postprocess import postprocess
from .serve_gate import ServeGateResult, score_serve, snap_end

log = logging.getLogger("refine.run")

_ROOT = Path(__file__).resolve().parents[3]   # repo root


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else _ROOT / p


@dataclass
class WindowDebug:
    """Everything diagnostics needs to draw one window's timeline."""
    window: Pass1Window
    feats: Features
    p_live: np.ndarray
    state: np.ndarray
    segments: list[tuple[float, float]]           # Stage C segments (seconds)
    serve: list[ServeGateResult] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)


def _record(window: Pass1Window, fps: float, verdict: str,
            reason: str | None, start: float | None, end: float | None,
            mean_p: float, gate: float, cross10: float) -> dict:
    return {
        "pass1_window_id": window.id,
        "pass1_start": round(window.start_frame / fps, 3),
        "pass1_end": round(window.end_frame / fps, 3),
        "pass1_start_frame": window.start_frame,
        "pass1_end_frame": window.end_frame,
        "verdict": verdict,
        "rejection_reason": reason,
        "refined_start": None if start is None else round(start, 3),
        "refined_end": None if end is None else round(end, 3),
        "refined_start_frame": None if start is None else int(round(start * fps)),
        "refined_end_frame": None if end is None else int(round(end * fps)),
        "confidence": {"mean_p_live": round(mean_p, 4),
                       "serve_gate_score": round(gate, 4),
                       "net_crossings_per_10s": round(cross10, 3)},
        "splits": [],
    }


def refine_window(window: Pass1Window, td: TrackingData, cfg: RefineConfig,
                  classifier: LiveClassifier) -> WindowDebug:
    """Run stages A-D over one candidate window's tracking span."""
    feats = compute_features(td, cfg)
    p = classifier.predict_proba(feats)
    segs, state = postprocess(p, cfg)
    fps = td.fps
    dbg = WindowDebug(window=window, feats=feats, p_live=p, state=state,
                      segments=[(float(feats.t[s.start]),
                                 float(feats.t[s.end - 1])) for s in segs])
    span_lo, span_hi = float(feats.t[0]), float(feats.t[-1])

    if not segs:
        reason = "low_live_ratio" if float(p.max(initial=0.0)) < cfg.enter_thresh \
            else "too_short"
        dbg.records.append(_record(window, fps, "rejected", reason,
                                   None, None, float(p.mean()) if len(p) else 0.0,
                                   0.0, 0.0))
        return dbg

    accepted: list[dict] = []
    for si, (seg, (t_start, t_end)) in enumerate(zip(segs, dbg.segments)):
        mean_p = float(p[seg.start:seg.end].mean())
        # For the window's FIRST segment the serve sits near the window start,
        # which the smoothed rising edge can lag — search from there.
        search_from = window.start_frame / fps if si == 0 else None
        gate = score_serve(feats, t_start, t_end, cfg, search_from=search_from)
        dbg.serve.append(gate)
        n_cross = int(np.sum((feats.crossings_t >= t_start)
                             & (feats.crossings_t <= t_end)))
        dur = max(t_end - t_start, 1e-6)
        cross10 = n_cross / dur * 10.0

        if mean_p < cfg.mean_p_live_min:
            dbg.records.append(_record(window, fps, "rejected", "low_live_ratio",
                                       None, None, mean_p, gate.score, cross10))
            continue
        if gate.score < cfg.serve_gate_min:
            dbg.records.append(_record(window, fps, "rejected", "no_serve",
                                       None, None, mean_p, gate.score, cross10))
            continue

        # Start anchor: serve contact when found; else back off from the first
        # net crossing by a typical serve-flight time (the rising edge lags the
        # real start, a crossing never precedes the serve); else rising edge.
        if gate.contact_t is not None:
            anchor = gate.contact_t
        else:
            cross = feats.crossings_t[(feats.crossings_t >= t_start - cfg.serve_lookback_s)
                                      & (feats.crossings_t <= t_end)]
            anchor = min(t_start, float(cross[0]) - 0.9) if len(cross) else t_start
        start = max(span_lo, anchor - cfg.pre_roll_s)
        ev_t, _kind = snap_end(feats, max(t_start, start), t_end, cfg)
        end = min(span_hi, ev_t + cfg.post_roll_s)
        if end - start < cfg.min_rally_s:
            dbg.records.append(_record(window, fps, "rejected", "too_short",
                                       None, None, mean_p, gate.score, cross10))
            continue
        rec = _record(window, fps, "rally", None, start, end,
                      mean_p, gate.score, cross10)
        accepted.append(rec)
        dbg.records.append(rec)

    if len(accepted) > 1:   # window split into several rallies
        spans = [[r["refined_start"], r["refined_end"]] for r in accepted]
        for r in accepted:
            r["splits"] = spans
    return dbg


def refine_video(video_id: str, cfg: RefineConfig | None = None,
                 classifier: LiveClassifier | None = None,
                 p1_source: str = "pass1") -> tuple[list[dict], list[WindowDebug]]:
    """Refine every Pass 1 window of one clip. Returns (records, debug)."""
    cfg = cfg or RefineConfig()
    classifier = classifier or make_classifier(cfg)
    video = load_video(video_id)
    fps = float(video["fps"])
    last_frame = int(video["frame_count"]) - 1
    margin = int(round(cfg.window_margin_s * fps))

    t0 = time.perf_counter()
    records: list[dict] = []
    debugs: list[WindowDebug] = []
    for w in load_pass1_windows(video_id, p1_source):
        f0 = max(0, w.start_frame - margin)
        f1 = min(last_frame, w.end_frame + margin)
        td = load_tracking(video_id, f0, f1, fps=fps,
                           reconstruct_3d=cfg.use_ballistic_3d,
                           max_rms_px=cfg.ballistic_max_rms_px)
        dbg = refine_window(w, td, cfg, classifier)
        records.extend(dbg.records)
        debugs.append(dbg)
    log.info("refine %s: %d windows -> %d records (%d rallies) in %.0f ms",
             video_id, len(debugs), len(records),
             sum(r["verdict"] == "rally" for r in records),
             (time.perf_counter() - t0) * 1e3)
    return records, debugs


def run_refine_json(video_id: str, cfg: RefineConfig | None = None,
                    plot: bool = False) -> Path:
    """Refine one clip and write ``<out_dir>/<video_id>.json`` plus the
    per-segment confidence log (JSONL, feeds the human-review queue)."""
    cfg = cfg or RefineConfig()
    records, debugs = refine_video(video_id, cfg)
    out_dir = resolve_path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{video_id}.json"
    out.write_text(json.dumps({"video_id": video_id, "records": records},
                              indent=2))
    from .diagnostics import write_confidence_log
    write_confidence_log(out_dir / f"{video_id}.segments.jsonl",
                         video_id, records)
    if plot:
        from .diagnostics import plot_clip
        plot_clip(video_id, debugs, cfg, out_dir / f"{video_id}.png")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Pass-2 rally refinement")
    ap.add_argument("video_id")
    ap.add_argument("--config", default=None, help="YAML config overrides")
    ap.add_argument("--classifier", choices=["rules", "xgboost"], default=None)
    ap.add_argument("--plot", action="store_true", help="write diagnostic PNG")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    cfg = load_config(args.config)
    if args.classifier:
        cfg = cfg.copy(classifier=args.classifier)
    out = run_refine_json(args.video_id, cfg, plot=args.plot)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
