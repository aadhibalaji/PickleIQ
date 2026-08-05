"""Match-stable player identity via appearance + court-position re-ID.

ByteTrack (players.PoseTracker) hands out per-frame `track_id`s that are
appearance-blind: they break whenever two players cross/occlude or a box is
dropped for a few frames. Because official numbers were pinned to `track_id`,
that is why the displayed player numbers kept switching mid-match.

This module keeps a small set of PERSISTENT identity slots — at most
`max_per_team` per team on this fixed camera (1 for singles, 2 for doubles;
see PoseTracker) — and assigns each frame's detections to them with a cost
that combines:
  * court-space distance (dominant; player motion is smooth frame-to-frame),
  * an HSV torso-colour histogram distance (breaks crossings/occlusions and
    re-acquires a player after a dropout or a between-rally gap), and
  * a dominant-colour (peak hue/sat bin) distance — a second, more specific
    read of the SAME crop/histogram, derived for free from the histogram
    already computed above (no extra detection or cv2 call). The full
    histogram is a diffuse shape signal; the dominant bin is closer to "what
    colour is this jersey", which is exactly the signal a same-team pair in
    similar-but-not-identical kit needs. See _dominant_color.

Identity survives ByteTrack id-switches AND window resets, so a player keeps the
same slot — and therefore the same number — across the whole match. The official
left->right numbering is still (re)anchored at each serve via
assign_numbers_at_serve(); this layer only decides WHICH physical player each
detection is, so that the number rides with the right body.

Deliberately lightweight (colour histogram, not a deep ReID embedding): the
roster is fixed at <=4 known players on a static camera, so this is all the
appearance signal needed and it runs on CPU with no extra model.

Same-kit doubles pairs (two players in near-identical colours) are the one
case colour genuinely cannot resolve. Rather than let position alone silently
pick a winner when it's also a near-tie (a real crossing/occlusion moment),
_match tracks the cost gap between the best assignment and the next-best
alternative; a small gap sets PlayerObs.identity_ambiguous=True on the
detections involved — an explicit "we're not sure which is which this frame"
flag, the same pattern rallyanalysis/coaching.py uses for low-confidence pose
metrics (flag, don't guess).
"""
from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger("cv.identity")

# Torso HSV histogram: Hue x Saturation, coarse bins. Value is dropped so the
# descriptor is (mostly) exposure-invariant across the frame.
_HIST_CHANNELS = [0, 1]
_HIST_SIZE = [16, 16]
_HIST_RANGES = [0, 180, 0, 256]


def _torso_hist(frame: np.ndarray, box, keypoints) -> np.ndarray | None:
    """Normalized Hue-Sat histogram of a player's torso patch, or None if the
    patch is too small. Prefers a tight crop from shoulder+hip keypoints (COCO
    5,6,11,12); falls back to the central band of the detection box (chest area,
    away from the head and legs)."""
    h, w = frame.shape[:2]
    x1 = y1 = x2 = y2 = None
    kp = keypoints
    if kp and len(kp) >= 13:
        pts = [kp[i] for i in (5, 6, 11, 12) if kp[i][0] > 0 and kp[i][1] > 0]
        if len(pts) >= 3:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
    if x1 is None:
        bx1, by1, bx2, by2 = box
        bw, bh = bx2 - bx1, by2 - by1
        x1, x2 = bx1 + 0.25 * bw, bx2 - 0.25 * bw       # central 50% horizontally
        y1, y2 = by1 + 0.20 * bh, by1 + 0.55 * bh       # chest band vertically
    xi1, yi1 = max(0, int(x1)), max(0, int(y1))
    xi2, yi2 = min(w, int(x2)), min(h, int(y2))
    if xi2 - xi1 < 4 or yi2 - yi1 < 4:
        return None
    hsv = cv2.cvtColor(frame[yi1:yi2, xi1:xi2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], _HIST_CHANNELS, None, _HIST_SIZE, _HIST_RANGES)
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist


# Bin widths for turning a peak (h_bin, s_bin) index back into an HSV value —
# derived from _HIST_SIZE/_HIST_RANGES above, not re-specified.
_HUE_BIN_W = (_HIST_RANGES[1] - _HIST_RANGES[0]) / _HIST_SIZE[0]   # 180/16
_SAT_BIN_W = (_HIST_RANGES[3] - _HIST_RANGES[2]) / _HIST_SIZE[1]   # 256/16
# A bin holding less than this fraction of the crop's total (normalized, L1)
# mass isn't a real dominant colour — it's a patterned/multi-colour garment or
# a noisy small crop, and the reading is untrustworthy. Uniform mass would be
# 1/256 =~ 0.0039 per bin; STARTING POINT, NOT CALIBRATED (same status as
# coaching.KP_CONF_MIN) — revisit once more clips are processed.
_MIN_DOMINANT_MASS = 0.05


def _dominant_color(hist: np.ndarray | None) -> tuple[float, float, float] | None:
    """(hue, sat, confidence) of the torso crop's single most common colour,
    read off the peak bin of the histogram _torso_hist already computed — no
    extra crop or cv2 call. confidence is that bin's share of the crop's
    total colour mass: high for a solid-colour jersey, low for a patterned/
    striped one or a crop too small to trust. None if the crop was too small
    for _torso_hist itself to return a histogram."""
    if hist is None:
        return None
    idx = int(np.argmax(hist))
    hi, si = idx // _HIST_SIZE[1], idx % _HIST_SIZE[1]
    hue = (hi + 0.5) * _HUE_BIN_W
    sat = (si + 0.5) * _SAT_BIN_W
    conf = float(hist[hi, si])
    return (hue, sat, conf)


def _hue_dist(a: float, b: float) -> float:
    """Circular distance between two OpenCV hues (period 180), normalized to
    [0, 1] — 0 = identical, 1 = maximally opposite (90 degrees apart)."""
    d = abs(a - b)
    return min(d, 180.0 - d) / 90.0


def _circular_ema(old: float, new: float, alpha: float, period: float = 180.0) -> float:
    """EMA for a circular quantity (hue): a naive linear average of e.g. 2 and
    178 would give ~90 (a totally different colour) instead of the true
    ~-0/180 boundary crossing. Steps by the shortest signed arc instead."""
    d = ((new - old + period / 2.0) % period) - period / 2.0
    return (old + alpha * d) % period


@dataclass
class _Identity:
    id: int
    team: str
    hist: np.ndarray | None = None
    dom_hue: float | None = None
    dom_sat: float | None = None
    dom_conf: float | None = None
    court_xy: tuple[float, float] | None = None
    last_frame: int = -1
    number: int | None = None


class PlayerIdentityTracker:
    """Persistent per-team identity slots. Call assign() every tracked frame;
    call new_window() at each Pass 2 window boundary and assign_numbers_at_serve()
    at each detected serve. Slots are never dropped, so ids stay in 1..4."""

    # cost = court_dist_m / POS_SCALE + APP_W * bhattacharyya(hist, hist)
    #        + COLOR_W * dominant_hue/sat distance (only when both readings
    #          are confident — see _MIN_DOMINANT_MASS)
    #        - hysteresis bonus for the identity tracked in nearby recent frames
    POS_SCALE = 2.0
    APP_W = 1.5
    # Deliberately smaller than APP_W: the dominant-colour term is correlated
    # with the full histogram (same crop), so it should sharpen a close call,
    # not dominate it or double-count the same evidence. STARTING POINT, NOT
    # CALIBRATED — same status as APP_W/KP_CONF_MIN.
    COLOR_W = 0.5
    MATCH_MAX = 2.2       # above this a detection prefers a NEW slot (if room)
    HIST_ALPHA = 0.12     # EMA rate for the appearance descriptor
    # Cost-gap threshold below which the best assignment isn't decisively
    # better than the next-best alternative for a team-frame — see _match's
    # ambiguous return. Roughly half STICKY_BONUS: a gap this small is well
    # within what a single noisy reading could flip, so it's reported as
    # uncertain rather than silently picked. STARTING POINT, NOT CALIBRATED.
    AMBIGUITY_MARGIN = 0.3
    # Hysteresis: a single noisy torso-histogram reading (small crop, coarse
    # 16x16 HSV bins) can swing APP_W=1.5 cost units on ONE frame — enough to
    # outweigh a decisive multi-metre position gap and steal a continuously-
    # tracked player away to a stale identity for a frame or two (confirmed on
    # real data: singles clip 9ac7be49..., frame 10453 — the identity matched
    # one frame earlier at 0.4m away lost to one dormant for 1467 frames at
    # 3.1m away, on appearance alone). STICKY_BONUS rewards whichever identity
    # was matched in the last STICKY_WINDOW frames, decaying linearly to 0 —
    # short enough that a genuine return after a real occlusion (which takes
    # longer than a few frames to resolve) still gets matched on position/
    # appearance alone, unaffected by this bonus.
    STICKY_BONUS = 0.6
    STICKY_WINDOW = 10    # frames
    # Hard cap on real identity slots per team, for the WHOLE run (this tracker
    # lives once per run_pass2() call — see PoseTracker.__init__ / run_pass2,
    # it is never re-created per window). Once both slots for a team exist,
    # _assign_team's cost-rejection branch is unreachable (only fires while
    # len(idents) < max_per_team) so every later detection force-binds to the
    # nearer slot regardless of cost. A 3rd+ same-team detection in one frame
    # is therefore provably noise, not a real player re-ID failure, and is
    # left with slot=None (see _assign_team's "over capacity" branch) — the
    # caller (pass2.py) must not paper over that with a fabricated id.
    #
    # Was a hardcoded class constant (always 2 = doubles capacity), which is
    # why a singles clip could accumulate up to 4 persistent identities (2
    # per side) from any bystander/spectator/shadow that got a stable court
    # projection near the boundary — see docs audit, cv/pass2.py wiring.
    # Now an instance parameter driven by videos.match_type: 1 (singles —
    # one player per side, matching players.py's serve numbering docstring)
    # or 2 (doubles / legacy unset rows, unchanged default).
    def __init__(self, max_per_team: int = 2) -> None:
        self.MAX_PER_TEAM = max_per_team
        self._identities: list[_Identity] = []
        self._next_id = 1

    def _team_ids(self, team: str) -> list[_Identity]:
        return [idn for idn in self._identities if idn.team == team]

    def new_window(self) -> None:
        """Start a fresh Pass 2 window. Keep identities (id/number/appearance)
        but forget their last court position: across the between-rally gap players
        walk to new spots, so the first frame of the window must re-acquire on
        APPEARANCE alone (position cost = 0), after which positions re-lock."""
        for idn in self._identities:
            idn.court_xy = None

    def assign(self, frame: np.ndarray, obs: list, frame_idx: int) -> None:
        """Set o.slot and o.number on each obs that has a team + court position.
        Obs projected off-court (court_xy is None) are left unassigned, exactly
        as before."""
        for team in ("near", "far"):
            dets = [o for o in obs if o.team == team and o.court_xy is not None]
            if dets:
                self._assign_team(frame, dets, team, frame_idx)

    def _assign_team(self, frame, dets, team, frame_idx) -> None:
        idents = self._team_ids(team)
        hists = [_torso_hist(frame, o.box, o.keypoints) for o in dets]
        doms = [_dominant_color(h) for h in hists]
        matched: set[int] = set()
        pairs, ambiguous = self._match(dets, hists, doms, idents, frame_idx)
        for di, ii in pairs:
            # Reject a poor match only when we can still afford a new slot; once
            # both slots exist, always bind to the best (a 3rd det is noise).
            if (self._cost(dets[di], hists[di], doms[di], idents[ii], frame_idx) > self.MATCH_MAX
                    and len(idents) < self.MAX_PER_TEAM):
                continue
            self._update(idents[ii], dets[di], hists[di], doms[di], frame_idx)
            dets[di].slot = idents[ii].id
            dets[di].number = idents[ii].number
            # Same-team identity was a near-tie this frame (position AND
            # colour both failed to separate the candidates decisively) —
            # e.g. same-kit doubles partners crossing. Flag it rather than
            # let the pick look as confident as an unambiguous frame.
            dets[di].identity_ambiguous = ambiguous
            matched.add(di)
        for di, o in enumerate(dets):
            if di in matched:
                continue
            if len(self._team_ids(team)) < self.MAX_PER_TEAM:
                idn = _Identity(id=self._next_id, team=team)
                self._next_id += 1
                self._identities.append(idn)
                self._update(idn, o, hists[di], doms[di], frame_idx)
                o.slot = idn.id
                o.number = idn.number
            # else over capacity -> slot/number stay None (dropped as before)

    def _cost(self, o, hist, dom, idn: _Identity, frame_idx: int) -> float:
        pos = 0.0
        if idn.court_xy is not None:
            pos = math.hypot(o.court_xy[0] - idn.court_xy[0],
                             o.court_xy[1] - idn.court_xy[1]) / self.POS_SCALE
        app = 1.0
        if hist is not None and idn.hist is not None:
            app = float(cv2.compareHist(idn.hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        cost = pos + self.APP_W * app
        # Dominant-colour term: only contributes when BOTH this detection's
        # crop and the identity's smoothed reading clear the confidence
        # floor. Below it, the "colour" is noise (small crop, patterned
        # kit) — excluded, not guessed at, same as coaching._kp_xy.
        if (dom is not None and idn.dom_hue is not None
                and dom[2] >= _MIN_DOMINANT_MASS and idn.dom_conf >= _MIN_DOMINANT_MASS):
            hue_d = _hue_dist(dom[0], idn.dom_hue)
            sat_d = abs(dom[1] - idn.dom_sat) / 256.0
            cost += self.COLOR_W * (0.7 * hue_d + 0.3 * sat_d)
        if idn.last_frame >= 0:
            age = frame_idx - idn.last_frame
            if age < self.STICKY_WINDOW:
                cost -= self.STICKY_BONUS * (1.0 - age / self.STICKY_WINDOW)
        return cost

    def _match(self, dets, hists, doms, idents,
               frame_idx: int) -> tuple[list[tuple[int, int]], bool]:
        """Minimum-cost injective matching of dets <-> idents. N is tiny (<=2 per
        team), so brute-force over permutations of the smaller side is optimal
        and dependency-free. Also returns whether the winning assignment beat
        the next-best alternative by less than AMBIGUITY_MARGIN — with N<=2
        this alternative is exactly "the two identities swapped", i.e. the
        one alternative reading that matters for a same-team mix-up."""
        n_d, n_i = len(dets), len(idents)
        if not n_d or not n_i:
            return [], False
        cost = [[self._cost(dets[d], hists[d], doms[d], idents[i], frame_idx)
                 for i in range(n_i)] for d in range(n_d)]
        best: list[tuple[int, int]] | None = None
        best_c: float | None = None
        second_c: float | None = None
        if n_d <= n_i:
            perms = itertools.permutations(range(n_i), n_d)
        else:
            perms = itertools.permutations(range(n_d), n_i)
        for perm in perms:
            if n_d <= n_i:
                c = sum(cost[d][perm[d]] for d in range(n_d))
                pairs = [(d, perm[d]) for d in range(n_d)]
            else:
                c = sum(cost[perm[i]][i] for i in range(n_i))
                pairs = [(perm[i], i) for i in range(n_i)]
            if best_c is None or c < best_c:
                second_c = best_c
                best_c, best = c, pairs
            elif second_c is None or c < second_c:
                second_c = c
        ambiguous = second_c is not None and (second_c - best_c) < self.AMBIGUITY_MARGIN
        return best or [], ambiguous

    def _update(self, idn: _Identity, o, hist, dom, frame_idx: int) -> None:
        idn.court_xy = o.court_xy
        idn.last_frame = frame_idx
        if hist is not None:
            idn.hist = hist.copy() if idn.hist is None else (
                (1.0 - self.HIST_ALPHA) * idn.hist + self.HIST_ALPHA * hist)
        if dom is not None:
            hue, sat, conf = dom
            if idn.dom_hue is None:
                idn.dom_hue, idn.dom_sat, idn.dom_conf = hue, sat, conf
            else:
                idn.dom_hue = _circular_ema(idn.dom_hue, hue, self.HIST_ALPHA)
                idn.dom_sat = (1.0 - self.HIST_ALPHA) * idn.dom_sat + self.HIST_ALPHA * sat
                idn.dom_conf = (1.0 - self.HIST_ALPHA) * idn.dom_conf + self.HIST_ALPHA * conf

    def assign_numbers_at_serve(self, obs: list) -> None:
        """Anchor official numbers left->right per side (near 1,2 / far 3,4;
        singles: 1 near, 2 far), binding each number to the persistent slot so it
        rides with that player until the next serve re-anchors."""
        by_id = {idn.id: idn for idn in self._identities}
        placed = [o for o in obs if o.court_xy is not None and o.slot is not None]
        near = sorted((o for o in placed if o.team == "near"), key=lambda o: o.court_xy[0])
        far = sorted((o for o in placed if o.team == "far"), key=lambda o: o.court_xy[0])
        n = 1
        for o in near + far:
            idn = by_id.get(o.slot)
            if idn is not None:
                idn.number = n
                o.number = n
            n += 1
