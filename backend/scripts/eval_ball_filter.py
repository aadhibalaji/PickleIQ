"""Before/after harness for the Phase-1 ball filtering layer (cv/ball_filter).

Compares the ball pipeline WITHOUT the filter (baseline = permissive detector
+ trajectory selector, exactly the pre-filter Pass 2 path) against the
pipeline WITH it (velocity gate + player-box motion logic + arc RANSAC +
interpolation) on the frames inside HUMAN rally labels.

Runs offline from stored data: player boxes/keypoints come from the saved
Pass 2 overlays (both pipelines see identical player context), so no pose /
GPU work — only the classical ball detector touches pixels. Ball ground truth
is scarce (a handful of ball_corrections rows), so most metrics are
physics/consistency proxies:

PRIMARY metrics come from human ground-truth marks (the viewer's "Mark ground
truth" mode, stored in ball_corrections with event_type/in_out):

  pos err px  pixel distance predicted-vs-marked at marked frames (any type)
  bounce P/R  predicted bounce within +/-N frames and +/-M court-metres of a
              marked bounce = match; precision over predictions, recall over
              marks
  in/out acc  predicted in-court call vs the human call on matched bounces

Physics/consistency proxies (the only metrics when a clip has no marks):

  det%        rally frames with a ball output (filtered also reports +interp)
  physviol    consecutive-pick pairs implying > max ball speed (impossible)
  footpick    picks inside the bottom 25% of a player box (certain shoe FPs)
  maxgap      longest frame gap between picks inside a rally (continuity)
  events      legacy height-heuristic events vs arc-junction events

Usage (from backend/):
  .venv/Scripts/python.exe scripts/eval_ball_filter.py [--video ID ...]
      [--json PATH] [--params-version N] [--bounce-frame-tol 5]
      [--bounce-dist-m 1.0] [--pos-frame-tol 1]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import db  # noqa: E402
from app.cv import court, overlays  # noqa: E402
from app.cv.ball_coarse import MotionBallDetector  # noqa: E402
from app.cv.ball_filter import (BallFrameFilter, classify_arc_events,  # noqa: E402
                                refine_track, _scale_m_per_px, MPH_TO_MPS)
from app.cv.ball_track import TrajectoryBallSelector  # noqa: E402
from app.cv.bounce import detect_events, reconstruct_world  # noqa: E402
from app.cv.kalman import smooth_track  # noqa: E402
from app.cv.pass2 import _in_court_region  # noqa: E402
from app.cv.video_io import iter_frames  # noqa: E402
from app.learning import params as learned_params  # noqa: E402
from app.pipeline import effective_calibration  # noqa: E402


def pick_videos(explicit: list[str]) -> list[dict]:
    """Videos with ground truth (GT marks and/or human rally labels) + stored
    pass2 overlays, deduped by filename (several DB rows point at the same
    clip file) — preferring the row with the most GT marks, since mark-based
    metrics are the primary ones."""
    with db.connect() as con:
        rows = con.execute(
            "SELECT v.id, v.filename, v.path, v.fps FROM videos v"
            " WHERE EXISTS (SELECT 1 FROM rally_labels r WHERE r.video_id=v.id"
            "               AND r.source='human')"
            "    OR EXISTS (SELECT 1 FROM ball_corrections b WHERE b.video_id=v.id)"
        ).fetchall()
        out = []
        for r in rows:
            if explicit and r["id"] not in explicit:
                continue
            n_ov = con.execute(
                "SELECT COUNT(*) c FROM overlays WHERE video_id=? AND source='pass2'",
                (r["id"],)).fetchone()["c"]
            n_marks = con.execute(
                "SELECT COUNT(*) c FROM ball_corrections WHERE video_id=?",
                (r["id"],)).fetchone()["c"]
            n_human = con.execute(
                "SELECT COUNT(*) c FROM rally_labels WHERE video_id=?"
                " AND source='human'", (r["id"],)).fetchone()["c"]
            if n_ov == 0 or not Path(r["path"]).exists():
                continue
            out.append({**dict(r), "n_overlay_chunks": n_ov,
                        "n_marks": n_marks, "n_human": n_human})
    best: dict[str, dict] = {}
    for v in out:
        cur = best.get(v["filename"])
        key = (v["n_marks"], v["n_human"], v["n_overlay_chunks"])
        if cur is None or key > (cur["n_marks"], cur["n_human"], cur["n_overlay_chunks"]):
            best[v["filename"]] = v
    return list(best.values())


def overlay_context(video_id: str, ws: int, we: int) -> dict[int, dict]:
    """frame -> {boxes: [(x1,y1,x2,y2)], feet: [(x,y)], keys: [id], cxy: []}"""
    ctx: dict[int, dict] = {}
    for e in overlays.read_range(video_id, "pass2", ws, we):
        boxes, feet, keys, cxy = [], [], [], []
        for p in e.get("players", []):
            boxes.append(tuple(p["box"]))
            keys.append(p.get("id"))
            if p.get("cxy"):
                cxy.append(tuple(p["cxy"]))
            for idx in (13, 14, 15, 16):
                kp = p.get("kp")
                if kp and len(kp) > idx and kp[idx][0] > 0 and kp[idx][1] > 0:
                    feet.append((float(kp[idx][0]), float(kp[idx][1])))
        ctx[e["f"]] = {"boxes": boxes, "feet": feet, "keys": keys, "cxy": cxy}
    return ctx


def in_foot_zone(x: float, y: float, boxes, frac: float = 0.25) -> bool:
    return any(x1 <= x <= x2 and y > y2 - frac * (y2 - y1)
               for x1, y1, x2, y2 in boxes)


def physics_violations(track: list[tuple], H, fps: float, vmax_mps: float,
                       max_gap: int = 15) -> tuple[int, int]:
    """(violations, pairs): consecutive output points implying > vmax_mps."""
    viol = pairs = 0
    for (f0, x0, y0), (f1, x1, y1) in zip(track, track[1:]):
        gap = f1 - f0
        if not (0 < gap <= max_gap):
            continue
        pairs += 1
        d_px = math.hypot(x1 - x0, y1 - y0)
        if d_px < 2:
            continue
        s = _scale_m_per_px(H, (x0 + x1) / 2, (y0 + y1) / 2)
        if d_px * s > vmax_mps * gap / fps:
            viol += 1
    return viol, pairs


def gap_stats(frames_out: list[int], ws: int, we: int) -> list[int]:
    """Frame gaps between successive picks inside [ws, we] (incl. lead-in/out
    to the window edges only if there are picks at all)."""
    fs = [f for f in frames_out if ws <= f <= we]
    return [b - a for a, b in zip(fs, fs[1:])] if len(fs) >= 2 else []


def load_marks(vid: str, H) -> list[dict]:
    """Ground-truth marks; tolerant of the pre-migration schema (rows then
    count as plain position marks). Court coords are filled from the current
    homography when the row predates it."""
    with db.connect() as con:
        rows = con.execute("SELECT * FROM ball_corrections WHERE video_id=?",
                           (vid,)).fetchall()
    marks = []
    for r in rows:
        k = r.keys()
        mk = {"frame": r["frame"], "x": r["x"], "y": r["y"],
              "event_type": (r["event_type"] if "event_type" in k and r["event_type"]
                             else "position"),
              "court_x": r["court_x"] if "court_x" in k else None,
              "court_y": r["court_y"] if "court_y" in k else None,
              "in_out": r["in_out"] if "in_out" in k else None,
              "disagree": (r["disagree"] if "disagree" in k else 0) or 0}
        if mk["court_x"] is None:
            cxy = court.img_to_court(H, np.array([[mk["x"], mk["y"]]]))[0]
            mk["court_x"], mk["court_y"] = float(cxy[0]), float(cxy[1])
        marks.append(mk)
    return marks


def match_bounces(gt: list[dict], pred: list[tuple], frame_tol: int,
                  dist_m: float) -> list[tuple[dict, tuple]]:
    """Greedy one-to-one matching of GT bounce marks to predicted bounce
    events (frame, court_x, court_y, in_court, kind), nearest frame first."""
    pairs = []
    used: set[int] = set()
    for g in sorted(gt, key=lambda g: g["frame"]):
        best = None
        for i, p in enumerate(pred):
            if i in used or abs(p[0] - g["frame"]) > frame_tol:
                continue
            d = math.hypot(p[1] - g["court_x"], p[2] - g["court_y"])
            if d > dist_m:
                continue
            score = (abs(p[0] - g["frame"]), d)
            if best is None or score < best[0]:
                best = (score, i)
        if best is not None:
            used.add(best[1])
            pairs.append((g, pred[best[1]]))
    return pairs


# ---------------------------------------------------------------------------
# --diagnose: stage attribution, condition slicing, failure galleries
# ---------------------------------------------------------------------------

NEAR_TOL = 25.0  # px: "on the ball"


def _pt_seg_dist(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / L2))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


def _court_lines_img(H):
    W, L, NY, K = court.COURT_W, court.COURT_L, court.NET_Y, court.KITCHEN
    lines = [[(0, 0), (W, 0)], [(0, L), (W, L)], [(0, 0), (0, L)], [(W, 0), (W, L)],
             [(0, NY), (W, NY)], [(0, NY - K), (W, NY - K)], [(0, NY + K), (W, NY + K)],
             [(W / 2, 0), (W / 2, NY - K)], [(W / 2, NY + K), (W / 2, L)]]
    out = []
    for a, b in lines:
        px = court.court_to_img(H, np.array([a, b], dtype=np.float64))
        out.append((px[0][0], px[0][1], px[1][0], px[1][1]))
    return out


def _box_dist(x, y, boxes):
    """0 when inside a box, else min pixel distance to the nearest box edge;
    inf when no boxes."""
    best = float("inf")
    for x1, y1, x2, y2 in boxes:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return 0.0
        dx = max(x1 - x, 0.0, x - x2)
        dy = max(y1 - y, 0.0, y - y2)
        best = min(best, math.hypot(dx, dy))
    return best


def _slice_keys(mk, marks, boxes, lines_img):
    side = "near" if mk["court_y"] >= court.NET_Y else "far"
    speed = "unknown"
    others = [o for o in marks if o is not mk and 0 < abs(o["frame"] - mk["frame"]) <= 15]
    if others:
        o = min(others, key=lambda o: abs(o["frame"] - mk["frame"]))
        s = math.hypot(o["x"] - mk["x"], o["y"] - mk["y"]) / abs(o["frame"] - mk["frame"])
        speed = "slow<5" if s < 5 else ("med5-15" if s <= 15 else "fast>15")
    bd = _box_dist(mk["x"], mk["y"], boxes)
    player = "inside" if bd == 0 else ("near<50px" if bd < 50 else "open")
    ld = min((_pt_seg_dist(mk["x"], mk["y"], *seg) for seg in lines_img), default=1e9)
    line = "line<30px" if ld < 30 else "open"
    return {"side": side, "speed": speed, "player": player, "line": line,
            "type": mk["event_type"]}


def _attribute_mark(mk, diag, filt_final, tn_mode):
    """Walk backwards through the pipeline for one mark and name where the
    ball was lost. Returns (category, detail)."""
    f = mk["frame"]

    def near(pts):
        best = None
        for df in (-1, 0, 1):
            for p in pts.get(f + df, []):
                e = math.hypot(p[0] - mk["x"], p[1] - mk["y"])
                if best is None or e < best[0]:
                    best = (e, p)
        return best

    fin = None
    for df in (-1, 0, 1):
        p = filt_final.get(f + df)
        if p:
            e = math.hypot(p[0] - mk["x"], p[1] - mk["y"])
            if fin is None or e < fin[0]:
                fin = (e, p)
    if fin and fin[0] <= NEAR_TOL:
        return None  # success — no failure to attribute

    raw = near(diag["raw"])
    good_raw = raw and raw[0] <= NEAR_TOL
    if not good_raw:
        if tn_mode:
            pk = near(diag["peak"])
            if pk and pk[0] <= NEAR_TOL:
                return ("DETECTOR_SUBTHRESHOLD", f"peak conf {pk[1][2]:.2f}")
        return ("DETECTOR_MISS", "no raw candidate within 25px")

    # A good raw candidate existed. Did the frame filter kill it?
    killed = None
    for df in (-1, 0, 1):
        for (x, y, reason) in diag["reasons"].get(f + df, []):
            if reason != "kept" and math.hypot(x - mk["x"], y - mk["y"]) <= NEAR_TOL:
                killed = reason
    if killed:
        return (f"FILTER_KILL:{killed}", "")
    dropped = near(diag["dropped"])
    if dropped and dropped[0] <= NEAR_TOL:
        return (f"ARC_DROP:{dropped[1][2]}", "")
    if fin and fin[1][2]:
        return ("INTERP_ERROR", f"interp {fin[0]:.0f}px off")
    if not tn_mode:
        picked = near(diag["pick"])
        if picked:
            return ("SELECTION_MISS",
                    f"selector took a blob {picked[0]:.0f}px away" if picked[0] > NEAR_TOL
                    else "picked then lost downstream")
        return ("SELECTION_NONE", "good candidate kept but selector chose nothing")
    return ("UNATTRIBUTED", "")


def _attribute_bounce_miss(mk, filt_final, arcs, params):
    f = mk["frame"]
    if not any(filt_final.get(f + d) for d in range(-3, 4)):
        return ("BOUNCE_TRACK_HOLE", "no track within ±3f")
    spanning = [a for a in arcs if a.f0 <= f <= a.f1]
    if spanning:
        a = spanning[0]
        if a.cy[0] >= 0:
            return ("BOUNCE_SWALLOWED:no_vertex", "arc y-curve never peaks")
        tv = -a.cy[1] / (2 * a.cy[0])
        span = a.f1 - a.f0
        if not (1.5 <= tv <= span - 1.5):
            return ("BOUNCE_SWALLOWED:vertex_at_edge", f"vtx t={tv:.1f} span={span}")
        return ("BOUNCE_SWALLOWED:vertex_shallow", "depth under threshold")
    prev = max((a for a in arcs if a.f1 < f), key=lambda a: a.f1, default=None)
    nxt = min((a for a in arcs if a.f0 > f), key=lambda a: a.f0, default=None)
    if prev and nxt:
        gap = nxt.f0 - prev.f1
        jg = int(params["ball_arc_join_gap_frames"])
        if gap > jg:
            return ("BOUNCE_JUNCTION_GAP", f"arc gap {gap}f > join {jg}f")
        return ("BOUNCE_JUNCTION_REJECTED", "junction existed (angle/hit class)")
    return ("BOUNCE_NO_ADJACENT_ARC", "track on one side only")


def _save_galleries(v, failures, diag, filt_final, out_dir):
    """Annotated crops for the worst 10 failures per category."""
    import cv2
    by_cat = defaultdict(list)
    for mk, cat, detail in failures:
        by_cat[cat].append(mk)
    wanted = {}
    for cat, mks in by_cat.items():
        for mk in mks[:10]:
            wanted.setdefault(mk["frame"], []).append((mk, cat))
    if not wanted:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_sorted = sorted(wanted)
    for lo, hi in [(frames_sorted[0], frames_sorted[-1])]:
        for f, frame in iter_frames(v["path"], start_frame=max(0, lo),
                                    end_frame=hi + 1):
            if f not in wanted:
                continue
            for mk, cat in wanted[f]:
                x0, y0 = int(mk["x"]), int(mk["y"])
                h, w = frame.shape[:2]
                cx0, cy0 = max(0, x0 - 170), max(0, y0 - 170)
                crop = frame[cy0:min(h, y0 + 170), cx0:min(w, x0 + 170)].copy()

                def draw(px, py, color, r=8, thick=2):
                    cv2.circle(crop, (int(px - cx0), int(py - cy0)), r, color, thick)

                draw(mk["x"], mk["y"], (0, 220, 255), 12, 3)  # GT: yellow
                for df in (-1, 0, 1):
                    for p in diag["raw"].get(f + df, []):
                        draw(p[0], p[1], (255, 255, 255), 5, 1)
                    for (x, y, reason) in diag["reasons"].get(f + df, []):
                        if reason != "kept":
                            draw(x, y, (0, 0, 255), 7, 1)
                    pk = diag["peak"].get(f + df)
                    if pk:
                        for p in pk:
                            draw(p[0], p[1], (160, 160, 160), 9, 1)
                fin = filt_final.get(f)
                if fin:
                    draw(fin[0], fin[1], (0, 255, 0), 4, 2)
                cv2.putText(crop, f"{cat} f{f}", (4, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                safe = cat.replace(":", "_")
                cv2.imwrite(str(out_dir / f"{safe}_f{f}.jpg"), crop)
    return out_dir


def eval_video(v: dict, params: dict, args) -> dict | None:
    vid = v["id"]
    H, cam, _ = effective_calibration(vid)
    if H is None or cam is None:
        print(f"  !! {v['filename']}: no calibration, skipped")
        return None
    tn = None
    if getattr(args, "detector", "classical") == "tracknet":
        from app.cv.ball_tracknet import TrackNetBall
        tn = TrackNetBall(weights=args.tracknet_weights)
        if not tn.available:
            print(f"  !! TrackNet weights not found at {args.tracknet_weights}")
            return None
    with db.connect() as con:
        human = [(r["start_frame"], r["end_frame"]) for r in con.execute(
            "SELECT start_frame, end_frame FROM rally_labels"
            " WHERE video_id=? AND source='human' ORDER BY start_frame", (vid,))]
        # No human rally labels (e.g. a re-upload marked directly with GT):
        # evaluate inside the pass1 windows instead — that is where Pass 2
        # runs detection anyway, so the comparison stays like-for-like.
        win_source = "human"
        if not human:
            human = [(r["start_frame"], r["end_frame"]) for r in con.execute(
                "SELECT start_frame, end_frame FROM rally_labels"
                " WHERE video_id=? AND source='pass1' ORDER BY start_frame", (vid,))]
            win_source = "pass1"
    if not human:
        print(f"  !! {v['filename']}: no rally windows (human or pass1), skipped")
        return None
    marks = load_marks(vid, H)
    fps = float(v["fps"] or 30.0)
    vmax = float(params["ball_max_speed_mph"]) * MPH_TO_MPS \
        * float(params["ball_speed_headroom"])

    def new_selector():
        return TrajectoryBallSelector(
            fps, gate_px=float(params["ball_gate_px"]),
            accel_px=float(params["ball_accel_px"]),
            foot_px=float(params["ball_foot_suppress_px"]),
            hit_speed_px=float(params["ball_hit_speed_px"]),
            acq_px=float(params["ball_acq_px"]),
            seed_min_px=float(params["ball_seed_min_px"]))

    m = {
        "video": v["filename"], "video_id": vid, "windows": len(human),
        "win_source": win_source,
        "frames": 0, "overlay_cov": 0,
        "raw_cands": 0,
        "base": {"det": 0, "foot": 0, "gaps": [], "viol": 0, "pairs": 0,
                 "events": defaultdict(int)},
        "filt": {"det": 0, "interp": 0, "foot": 0, "gaps": [], "viol": 0,
                 "pairs": 0, "events": defaultdict(int),
                 "rej": {"velocity": 0, "foot_zone": 0, "persistent": 0},
                 "outliers": 0, "fragments": 0, "arcs": 0},
    }

    base_out_all: list[tuple] = []   # (f, x, y) final outputs across windows
    filt_out_all: list[tuple] = []
    events_b: list[tuple] = []       # (frame, court_x, court_y, in_court, kind)
    events_f: list[tuple] = []
    tn_raw: dict[int, tuple] = {}    # raw TrackNet dets (pre-filter), parity stat

    diagnose = bool(getattr(args, "diagnose", False))
    diag = {"raw": {}, "reasons": {}, "pick": {}, "peak": {}, "dropped": {}}
    near_mark: set[int] = set()
    if diagnose:
        for mk in marks:
            near_mark.update((mk["frame"] - 1, mk["frame"], mk["frame"] + 1))
    filt_final: dict[int, tuple] = {}   # frame -> (x, y, interpolated)
    arcs_all: list = []
    boxes_all: dict[int, list] = {}

    for ws, we in human:
        ctx = overlay_context(vid, ws, we)
        # Mirror Pass 2's detector construction EXACTLY (pass2.py): the harness
        # previously omitted motion_thresh/min_circularity, so it evaluated the
        # detector at the class defaults (thresh 12, circ 0.35) while production
        # ran the learned values (3, 0.30) — i.e. it wasn't measuring what ships.
        det_b = MotionBallDetector(
            min_area=2.0, max_area=600.0,
            sat_min=float(params["ball_sat_min"]), val_min=float(params["ball_val_min"]),
            hue_lo=float(params["ball_hue_lo"]), hue_hi=float(params["ball_hue_hi"]),
            motion_thresh=int(params.get("ball_motion_thresh", 3.0)),
            min_circularity=float(params.get("ball_min_circularity", 0.30)))
        sel_b, sel_f = new_selector(), new_selector()
        ff = BallFrameFilter(
            H, fps, max_speed_mph=float(params["ball_max_speed_mph"]),
            headroom=float(params["ball_speed_headroom"]),
            persist_frames=int(params["ball_box_persist_frames"]),
            persist_match_px=float(params["ball_box_persist_px"]),
            foot_zone_frac=float(params["ball_foot_zone_frac"]))
        obs_b: list[tuple] = []
        obs_f: list[tuple] = []
        players_by_frame: dict[int, list] = {}
        boxes_by_frame: dict[int, list] = {}
        tbuf: list = []

        for f, frame in iter_frames(v["path"], start_frame=ws, end_frame=we + 1):
            m["frames"] += 1
            c = ctx.get(f, {"boxes": [], "feet": [], "keys": [], "cxy": []})
            if c["boxes"]:
                m["overlay_cov"] += 1
            players_by_frame[f] = c["cxy"]
            boxes_by_frame[f] = c["boxes"]
            boxes_all[f] = c["boxes"]
            cands = [cd for cd in det_b.update(frame)
                     if _in_court_region(H, cd.x, cd.y)]
            m["raw_cands"] += len(cands)
            # -- baseline: classical selector on the raw pool (pre-filter
            #    Pass 2 path) — the fixed comparison anchor in BOTH modes
            pb = sel_b.select(f, cands, c["boxes"], c["feet"])
            if pb:
                obs_b.append((f, pb.x, pb.y, pb.diameter_px))
            keyed = list(zip(c["keys"], c["boxes"]))
            if tn is not None:
                # -- filtered pipeline with TrackNet as the detector: its one
                #    dense det per frame goes through filter stages 1-2 (the
                #    layer applies to any detector); no selector needed.
                tbuf.append(frame)
                if len(tbuf) > 3:
                    tbuf.pop(0)
                d = tn.infer_triplet(tbuf) if len(tbuf) == 3 else None
                if diagnose and f in near_mark and len(tbuf) == 3 and tn.last_peak:
                    diag["peak"][f] = [tn.last_peak]
                if d is not None:
                    tn_raw[f] = (d[0], d[1])
                    if diagnose and f in near_mark:
                        diag["raw"][f] = [(d[0], d[1])]
                    tc = SimpleNamespace(x=d[0], y=d[1], diameter_px=d[2], score=d[3])
                    kept_tc = (_in_court_region(H, tc.x, tc.y)
                               and ff.filter(f, [tc], keyed))
                    if diagnose and f in near_mark:
                        diag["reasons"][f] = [(c2.x, c2.y, r2)
                                              for c2, r2 in getattr(ff, "last_reasons", [])]
                    if kept_tc:
                        obs_f.append((f, tc.x, tc.y, tc.diameter_px))
                        ff.confirm(f, tc.x, tc.y)
            else:
                # -- filtered: stages 1-2, then the same selector
                fc = ff.filter(f, cands, keyed)
                pf = sel_f.select(f, fc, c["boxes"], c["feet"])
                if diagnose and f in near_mark:
                    diag["raw"][f] = [(cd.x, cd.y) for cd in cands]
                    diag["reasons"][f] = [(c2.x, c2.y, r2)
                                          for c2, r2 in getattr(ff, "last_reasons", [])]
                    if pf:
                        diag["pick"][f] = [(pf.x, pf.y)]
                if pf:
                    obs_f.append((f, pf.x, pf.y, pf.diameter_px))
                    ff.confirm(f, pf.x, pf.y)

        # ---- batch stage (filtered only) + final outputs ------------------
        refined = refine_track(
            obs_f, split_deg=float(params["ball_arc_split_deg"]),
            min_speed_px=float(params["ball_arc_min_speed_px"]),
            gap_frames=int(params["ball_arc_gap_frames"]),
            tol_px=float(params["ball_arc_tol_px"]),
            min_arc_pts=int(params["ball_arc_min_pts"]),
            join_gap=int(params["ball_arc_join_gap_frames"]),
            split_dv_px=float(params["ball_arc_split_dv_px"]),
            claim_tol_mult=float(params["ball_arc_claim_tol_mult"]),
            max_interp_gap=int(params["ball_arc_max_interp_gap"]))
        for k in ("velocity", "foot_zone", "persistent"):
            m["filt"]["rej"][k] += getattr(ff.stats, k)
        m["filt"]["outliers"] += refined.n_outliers
        m["filt"]["fragments"] += refined.n_fragments
        m["filt"]["arcs"] += len(refined.arcs)

        base_pts = [(f, x, y) for f, x, y, _ in obs_b]
        filt_pts = sorted([(p.frame, p.x, p.y) for p in refined.points]
                          + [(p.frame, p.x, p.y) for p in refined.interpolated])
        for p in refined.points:
            filt_final[p.frame] = (p.x, p.y, False)
        for p in refined.interpolated:
            filt_final.setdefault(p.frame, (p.x, p.y, True))
        arcs_all += refined.arcs
        if diagnose:
            for (pf_, px_, py_, reason) in refined.dropped:
                diag["dropped"].setdefault(pf_, []).append((px_, py_, reason))
        m["base"]["det"] += len(base_pts)
        m["filt"]["det"] += len(refined.points)
        m["filt"]["interp"] += len(refined.interpolated)
        m["base"]["gaps"] += gap_stats([p[0] for p in base_pts], ws, we)
        m["filt"]["gaps"] += gap_stats([p[0] for p in filt_pts], ws, we)
        for label, pts in (("base", base_pts), ("filt", filt_pts)):
            m[label]["foot"] += sum(
                1 for f, x, y in pts
                if in_foot_zone(x, y, boxes_by_frame.get(f, [])))
            viol, pairs = physics_violations(pts, H, fps, vmax)
            m[label]["viol"] += viol
            m[label]["pairs"] += pairs
        base_out_all += base_pts
        filt_out_all += filt_pts

        # ---- events -------------------------------------------------------
        # Baseline: legacy height-heuristic events on the Kalman/world track.
        # Mirrors Pass 2: size->depth honesty gate + homography-exact bounce
        # positions, so the harness measures what actually ships.
        track_b = smooth_track(obs_b, max_gap=int(fps * 0.4),
                               gate2=float(params["ball_kalman_gate2"]))
        world_b = reconstruct_world(
            track_b, cam,
            height_sigma_max=float(params.get("ball_height_max_sigma_m", 0.35)))
        for ev in detect_events(world_b, players_by_frame, params, fps, H=H,
                                ball_px={bf: (bx, by)
                                         for bf, bx, by, _ in obs_b}):
            m["base"]["events"][ev.kind + ("" if ev.in_court else "_out")] += 1
            events_b.append((ev.frame, ev.court_x, ev.court_y, ev.in_court, ev.kind))
        # Filtered: arc-junction events.
        for ev in classify_arc_events(
                refined.arcs, boxes_by_frame,
                join_gap=int(params["ball_arc_join_gap_frames"]),
                min_angle_deg=float(params["ball_event_min_angle_deg"]),
                adjacent_frac=float(params["ball_hit_adjacent_frac"]),
                vertex_min_px=float(params["ball_bounce_vertex_min_px"]),
                plausibility=params["ball_bounce_plausibility"] > 0.5,
                vy_frames=int(params["ball_bounce_vy_frames"]),
                vy_min_px=float(params["ball_bounce_vy_min_px"]),
                vy_trust_span=int(params["ball_bounce_vy_trust_span"]),
                merge_frames=int(params["ball_bounce_merge_frames"])):
            cxy = court.img_to_court(H, np.array([[ev.x, ev.y]]))[0]
            inc = bool(court.in_court(np.array([[cxy[0], cxy[1]]]))[0])
            m["filt"]["events"][ev.kind + ("" if inc else "_out")] += 1
            events_f.append((ev.frame, float(cxy[0]), float(cxy[1]), inc, ev.kind))

    # ---- ground-truth mark metrics (PRIMARY when marks exist) --------------
    def in_win(f):
        return any(ws <= f <= we for ws, we in human)

    gmarks = [mk for mk in marks if in_win(mk["frame"])]
    gtb = [mk for mk in gmarks if mk["event_type"] == "bounce"]
    m["gt"] = {
        "total": len(marks), "in_windows": len(gmarks),
        "bounce": len(gtb),
        "hit": sum(1 for mk in gmarks if mk["event_type"] == "hit"),
        "position": sum(1 for mk in gmarks if mk["event_type"] == "position"),
        "disagree": sum(mk["disagree"] for mk in gmarks),
    }
    for label, pts, evs in (("base", base_out_all, events_b),
                            ("filt", filt_out_all, events_f)):
        errs, miss = [], 0
        for mk in gmarks:
            near = [p for p in pts if abs(p[0] - mk["frame"]) <= args.pos_frame_tol]
            if near:
                p = min(near, key=lambda p: abs(p[0] - mk["frame"]))
                errs.append(round(math.hypot(p[1] - mk["x"], p[2] - mk["y"]), 1))
            else:
                miss += 1
        pred_b = [e for e in evs if e[4] == "bounce"]
        pairs = match_bounces(gtb, pred_b, args.bounce_frame_tol, args.bounce_dist_m)
        if label == "filt":
            filt_pairs, filt_pred_b = pairs, pred_b
        judged = [(g, p) for g, p in pairs if g["in_out"] in ("in", "out")]
        m[label]["pos_err"] = errs
        m[label]["pos_miss"] = miss
        m[label]["bounce_pred"] = len(pred_b)
        m[label]["bounce_matched"] = len(pairs)
        m[label]["inout_n"] = len(judged)
        m[label]["inout_ok"] = sum(
            1 for g, p in judged if (g["in_out"] == "in") == bool(p[3]))
    if diagnose:
        lines_img = _court_lines_img(H)
        # -- (a) stage attribution ------------------------------------------
        failures = []
        for mk in gmarks:
            r = _attribute_mark(mk, diag, filt_final, tn is not None)
            if r:
                failures.append((mk, r[0], r[1]))
        fail_counts = defaultdict(int)
        for _, cat, _ in failures:
            fail_counts[cat] += 1
        matched_gt = {id(g) for g, _ in filt_pairs}
        bounce_fail = defaultdict(int)
        for mk in gtb:
            if id(mk) not in matched_gt:
                cat, _ = _attribute_bounce_miss(mk, filt_final, arcs_all, params)
                bounce_fail[cat] += 1
        matched_pred = {p for _, p in filt_pairs}
        fp_by_rally = defaultdict(int)
        for e in filt_pred_b:
            if e not in matched_pred:
                ri = next((i + 1 for i, (ws, we) in enumerate(human)
                           if ws <= e[0] <= we), 0)
                fp_by_rally[f"r{ri}"] += 1
        # -- (b) condition slicing ------------------------------------------
        def final_err(mk):
            best = None
            for df in (-1, 0, 1):
                p = filt_final.get(mk["frame"] + df)
                if p:
                    e = math.hypot(p[0] - mk["x"], p[1] - mk["y"])
                    best = e if best is None else min(best, e)
            return best
        errs_all = [e for mk in gmarks if (e := final_err(mk)) is not None]
        pooled_med = float(np.median(errs_all)) if errs_all else 0.0
        slice_rows = []
        for dim in ("side", "speed", "player", "line", "type"):
            groups = defaultdict(list)
            for mk in gmarks:
                keys = _slice_keys(mk, gmarks, boxes_all.get(mk["frame"], []), lines_img)
                groups[keys[dim]].append(mk)
            for val, mks in sorted(groups.items()):
                es = [e for mk in mks if (e := final_err(mk)) is not None]
                hit = sum(1 for e in es if e <= NEAR_TOL)
                med = float(np.median(es)) if es else float("nan")
                flag = ("*HIGH" if es and pooled_med > 0 and med > 2 * pooled_med else "")
                flag += ("~lowN" if len(mks) < 8 else "")
                slice_rows.append((dim, val, len(mks),
                                   100.0 * hit / max(1, len(mks)), med, flag))
        # -- (c) galleries ----------------------------------------------------
        from app.config import DATA_DIR
        gal = _save_galleries(
            v, failures, diag, filt_final,
            DATA_DIR / "eval_diag" / vid[:8] / getattr(args, "tag", "run"))
        # -- (d) report -------------------------------------------------------
        top = sorted(fail_counts.items(), key=lambda kv: -kv[1])
        levers = {
            "DETECTOR_MISS": "better detector coverage (TrackNet / fine-tune)",
            "DETECTOR_SUBTHRESHOLD": "lower TrackNet conf threshold (peaks exist)",
            "SELECTION_MISS": "selector association (gates/priors)",
            "SELECTION_NONE": "selector acquisition (cold start)",
            "INTERP_ERROR": "shrink ball_arc_max_interp_gap",
            "BOUNCE_TRACK_HOLE": "detector coverage at bounces",
        }
        rep = [f"  -- DIAGNOSE ({m['windows']} windows, filtered pipeline) --",
               f"  mark failures {len(failures)}/{len(gmarks)}:"]
        rep += [f"    {cat:32} {n:4}" for cat, n in top]
        if bounce_fail:
            rep.append(f"  bounce-event misses ({sum(bounce_fail.values())}/{len(gtb)} GT):")
            rep += [f"    {cat:32} {n:4}"
                    for cat, n in sorted(bounce_fail.items(), key=lambda kv: -kv[1])]
        if fp_by_rally:
            rep.append("  unmatched predicted bounces by rally (for exhaustive-"
                       "window precision): "
                       + " ".join(f"{k}:{n}" for k, n in sorted(fp_by_rally.items())))
        rep.append(f"  slices (n / on-ball% / med err; pooled med {pooled_med:.1f}px):")
        for dim, val, n, hitp, med, flag in slice_rows:
            rep.append(f"    {dim:6} {val:10} n={n:3}  {hitp:5.1f}%  "
                       f"{med:7.1f}px  {flag}")
        if gal:
            rep.append(f"  galleries: {gal}")
        rep.append("  top-3 leverage: " + "; ".join(
            f"{cat} ({n}) -> {levers.get(cat.split(':')[0], 'inspect gallery')}"
            for cat, n in top[:3]))
        m["diag_report"] = "\n".join(rep)
        m["diag"] = {"failures": dict(fail_counts),
                     "bounce_failures": dict(bounce_fail),
                     "fp_bounces_by_rally": dict(fp_by_rally),
                     "slices": slice_rows,
                     "pooled_med_px": pooled_med,
                     # every predicted bounce with its rally + matched flag, so
                     # precision can be recomputed over any window subset (the
                     # exhaustively-marked rallies give TRUE false positives)
                     "filt_bounce_events": [
                         [e[0], bool(e[3]), e in matched_pred,
                          next((i + 1 for i, (ws, we) in enumerate(human)
                                if ws <= e[0] <= we), 0)]
                         for e in filt_pred_b]}
    if tn is not None:
        # Raw detector recall@25px vs marks (near/far) — the acceptance-test
        # stat, here confirming FP16/OpenVINO parity and detector quality on
        # any new clip before the pipeline layers on top.
        def raw_seen(mk):
            return any(
                math.hypot(p[0] - mk["x"], p[1] - mk["y"]) <= 25.0
                for df in (-1, 0, 1) if (p := tn_raw.get(mk["frame"] + df)))
        nearm = [mk for mk in gmarks if mk["court_y"] >= court.NET_Y]
        farm = [mk for mk in gmarks if mk["court_y"] < court.NET_Y]
        m["tn_recall"] = {
            "near": [sum(1 for mk in nearm if raw_seen(mk)), len(nearm)],
            "far": [sum(1 for mk in farm if raw_seen(mk)), len(farm)],
        }
    return m


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
    ap.add_argument("--tracknet-weights",
                    default=str(Path(__file__).resolve().parents[2]
                                / "data" / "models" / "wasb_tracknetv2_tennis.pth.tar"),
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

    if args.params_version:
        params = learned_params.get_version(args.params_version)
    else:
        _, params = learned_params.get_latest()
    for kv in args.set:
        k, _, val = kv.partition("=")
        params[k.strip()] = float(val)
        print(f"override: {k.strip()} = {val}")

    videos = pick_videos(args.video)
    if not videos:
        print("No evaluable videos (need human rally labels + pass2 overlays "
              "+ the clip file on disk).")
        return
    print(f"Evaluating {len(videos)} clip(s): baseline (no filter) vs "
          f"Phase-1 filtering layer\n")
    results = []
    for v in videos:
        m = eval_video(v, params, args)
        if m:
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
        Path(args.json).write_text(json.dumps(results, indent=2, default=str))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
