"""In-app assistant: a Claude conversation with full context of the current
clip that can DRIVE THE APP through a bounded, safe tool surface.

Design boundary (deliberate): this assistant can read everything (rallies,
discrepancies, per-frame feature traces, params, notes) and take the same
actions the UI exposes (tune clamped parameters, trigger reprocessing, write
notes). It CANNOT edit the CV source code — that stays with the coding agent
in the terminal. When a real code change is needed the assistant drafts a
precise handoff instead of pretending it changed code.

The server runs the whole tool loop per HTTP turn and returns only the final
assistant text plus a list of actions taken; the client keeps a simple text
history. Current clip state is re-embedded in the system prompt every turn,
so the assistant always sees fresh state without round-tripping tool blocks.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import db, pipeline
from ..analysis.metrics import Rally, compare
from ..config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL
from ..cv import overlays
from ..learning import params as learned_params

router = APIRouter()
log = logging.getLogger("api.chat")


class ChatMessage(BaseModel):
    role: str
    text: str


class ChatRequest(BaseModel):
    video_id: str
    messages: list[ChatMessage]


# --------------------------------------------------------------------- tools
TOOLS = [
    {
        "name": "get_feature_trace",
        "description": "Per-frame Pass 2 signals (ball pixel xy+diameter, "
                       "reconstructed court x/y/height, player boxes+skeletons, "
                       "detected bounce/hit events, player speed) for a frame "
                       "range. This is how you diagnose a specific moment the "
                       "user points at — e.g. why a ball was called out, or "
                       "whether the far ball was tracked. Keep ranges small "
                       "(<= ~120 frames) or it is downsampled.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_frame": {"type": "integer"},
                "end_frame": {"type": "integer"},
            },
            "required": ["start_frame", "end_frame"],
        },
    },
    {
        "name": "list_rallies",
        "description": "All rally boundaries: human labels, Pass 1 candidates, "
                       "and Pass 2 detections (with each Pass 2 rally's end "
                       "reason and confirmation).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "compare_to_human",
        "description": "Score Pass 2 against the human labels: temporal IoU, "
                       "per-rally start/end errors, and missed/extra rallies. "
                       "Requires human labels to exist.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "apply_params",
        "description": "Update global LearnedParams (inherited by ALL future "
                       "clips). Values are clamped to safe ranges. Use for "
                       "method-level tuning, e.g. out_max_dist_m, "
                       "bounce_max_height_m, end_offset_frames, p1_motion_ratio. "
                       "Explain the change to the user first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "updates": {
                    "type": "object",
                    "description": "param name -> new numeric value",
                    "additionalProperties": {"type": "number"},
                },
                "reason": {"type": "string"},
            },
            "required": ["updates", "reason"],
        },
    },
    {
        "name": "reprocess",
        "description": "Re-run detection with current params/calibration. "
                       "'pass1' re-runs the coarse gate then Pass 2; 'pass2' "
                       "re-runs only the fine boundary detection. Overlays "
                       "regenerate from metadata; no new video is stored.",
        "input_schema": {
            "type": "object",
            "properties": {"which": {"type": "string", "enum": ["pass1", "pass2"]}},
            "required": ["which"],
        },
    },
    {
        "name": "add_note",
        "description": "Save a note on this clip (feeds the discrepancy report "
                       "and the coding-agent handoff). Use this to record a "
                       "precise code-change request when a fix needs source "
                       "edits you cannot make.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "scope": {"type": "string", "enum": ["global", "clip_local"]},
            },
            "required": ["text"],
        },
    },
]


def _tool_get_feature_trace(video_id: str, a: int, b: int) -> str:
    a, b = int(a), int(b)
    if b - a > 240:
        b = a + 240
    ents = overlays.read_range(video_id, "pass2", a, b)
    if not ents:
        return db.dumps({"note": "no Pass 2 metadata in this range — the range "
                                 "may be outside a rally window, or Pass 2 has "
                                 "not run.", "frames": []})
    keep = ["f", "ball", "world", "evt", "pspeed", "obs"]
    step = max(1, len(ents) // 80)
    slim = [{k: e[k] for k in keep if k in e} for e in ents[::step]]
    return db.dumps({"frames": slim, "legend": {
        "ball": "[x_px, y_px, diameter_px, confidence]",
        "world": "[court_x_m, court_y_m, height_m|null]  court is 6.10 x 13.41 m,"
                 " net at y=6.71, far side y<6.71",
        "evt": "detected bounce/hit {kind,in,side,cx,cy}",
        "obs": "true=real detection, false=Kalman-bridged (untrustworthy)"}})


def _tool_list_rallies(video_id: str) -> str:
    with db.connect() as con:
        rows = db.rows_to_dicts(con.execute(
            "SELECT source,start_frame,end_frame,meta FROM rally_labels"
            " WHERE video_id=? ORDER BY source,start_frame", (video_id,)).fetchall())
    out: dict = {"human": [], "pass1": [], "pass2": []}
    for r in rows:
        item = {"start": r["start_frame"], "end": r["end_frame"]}
        if r["source"] == "pass2":
            item["meta"] = db.loads(r["meta"], {})
        out[r["source"]].append(item)
    return db.dumps(out)


def _tool_compare(video_id: str) -> str:
    with db.connect() as con:
        v = con.execute("SELECT fps FROM videos WHERE id=?", (video_id,)).fetchone()
        rows = db.rows_to_dicts(con.execute(
            "SELECT source,start_frame,end_frame FROM rally_labels WHERE video_id=?",
            (video_id,)).fetchall())
    human = [Rally(r["start_frame"], r["end_frame"]) for r in rows if r["source"] == "human"]
    ai = [Rally(r["start_frame"], r["end_frame"]) for r in rows if r["source"] == "pass2"]
    if not human:
        return db.dumps({"error": "no human labels yet"})
    return db.dumps(compare(human, ai, v["fps"] or 30.0))


def _tool_apply_params(updates: dict, reason: str) -> str:
    clean = {k: v for k, v in updates.items() if k in learned_params.DEFAULTS}
    unknown = [k for k in updates if k not in learned_params.DEFAULTS]
    if not clean:
        return db.dumps({"error": "no valid params", "unknown": unknown,
                         "valid_params": list(learned_params.DEFAULTS)})
    version = learned_params.apply_updates(clean, origin=f"chat: {reason[:80]}")
    _, current = learned_params.get_latest()
    return db.dumps({"applied_version": version,
                     "new_values": {k: current[k] for k in clean},
                     "ignored_unknown": unknown})


def _tool_reprocess(video_id: str, which: str) -> str:
    if which not in ("pass1", "pass2"):
        return db.dumps({"error": "which must be pass1 or pass2"})
    return db.dumps(pipeline.reprocess(video_id, which))


def _tool_add_note(video_id: str, text: str, scope: str) -> str:
    nid = db.new_id()
    with db.connect() as con:
        con.execute(
            "INSERT INTO user_notes (id, video_id, text, source, scope, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (nid, video_id, text, "typed", scope if scope in ("global", "clip_local") else "global",
             db.now()))
    return db.dumps({"saved_note_id": nid})


def _dispatch(video_id: str, name: str, inp: dict) -> str:
    try:
        if name == "get_feature_trace":
            return _tool_get_feature_trace(video_id, inp["start_frame"], inp["end_frame"])
        if name == "list_rallies":
            return _tool_list_rallies(video_id)
        if name == "compare_to_human":
            return _tool_compare(video_id)
        if name == "apply_params":
            return _tool_apply_params(inp["updates"], inp.get("reason", ""))
        if name == "reprocess":
            return _tool_reprocess(video_id, inp["which"])
        if name == "add_note":
            return _tool_add_note(video_id, inp["text"], inp.get("scope", "global"))
        return db.dumps({"error": f"unknown tool {name}"})
    except Exception as exc:  # noqa: BLE001 — tool errors must not break the loop
        log.warning("tool %s failed: %s", name, exc)
        return db.dumps({"error": f"{type(exc).__name__}: {exc}"})


def _state_snapshot(video_id: str) -> str:
    with db.connect() as con:
        v = con.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
        if v is None:
            return "video not found"
        counts = {s: con.execute(
            "SELECT COUNT(*) c FROM rally_labels WHERE video_id=? AND source=?",
            (video_id, s)).fetchone()["c"] for s in ("human", "pass1", "pass2")}
        cal = con.execute(
            "SELECT scope FROM calibrations WHERE (scope='global') OR id=("
            "SELECT calibration_id FROM videos WHERE id=?) ORDER BY created_at DESC LIMIT 1",
            (video_id,)).fetchone()
        notes = db.rows_to_dicts(con.execute(
            "SELECT text,scope FROM user_notes WHERE video_id=? ORDER BY created_at DESC LIMIT 20",
            (video_id,)).fetchall())
        report = con.execute(
            "SELECT summary FROM reports WHERE video_id=? ORDER BY created_at DESC LIMIT 1",
            (video_id,)).fetchone()
    version, params = learned_params.get_latest()
    fps = v["fps"] or 30.0
    lines = [
        f"CLIP: {v['filename']}  {v['width']}x{v['height']}  {fps:.1f}fps  "
        f"{v['frame_count']} frames ({v['frame_count']/fps:.0f}s)",
        f"Frame<->seconds: seconds = frame / {fps:.2f}",
        f"Rally counts: human={counts['human']}  pass1={counts['pass1']}  pass2={counts['pass2']}",
        f"Court calibration: {'yes ('+cal['scope']+')' if cal else 'NONE — Pass 2 cannot run'}",
        f"LearnedParams version {version} (global baseline).",
        "Current params: " + db.dumps(params),
        "Non-default params: " + db.dumps(
            {k: params[k] for k in params if params[k] != learned_params.DEFAULTS.get(k)}),
    ]
    if report:
        lines.append("Latest report summary: " + report["summary"])
    if notes:
        lines.append("Recent notes: " + db.dumps(notes))
    return "\n".join(lines)


SYSTEM = """You are the in-app assistant for a pickleball rally-detection system,
embedded in its web UI. You are the same model the user works with in their
terminal (Claude), so treat this as one continuous collaboration.

HOW THE SYSTEM WORKS (so your diagnoses are grounded):
- Two passes. Pass 1 is a cheap coarse gate over all footage; its live-play
  signal is the ball repeatedly crossing the net (within the calibrated court
  region). Pass 2 runs only inside padded Pass 1 windows: full-rate player
  tracking (YOLO11-pose + ByteTrack), ball tracking (classical fallback unless
  TrackNet weights are installed), Kalman smoothing, then 3D reconstruction —
  the ball's apparent pixel size gives depth, the homography gives the court
  surface, together they give height above the court. Bounces are height
  minima near zero away from players; the rally END is a terminal ball event
  (bounce out / double bounce / net fail) confirmed but never timed by player
  behaviour.
- Known weak spot: the far-side ball is only a few pixels wide, so its
  size->depth->court-position reconstruction is noisy. This is a MODEL-WEIGHTS
  limitation (fixed by installing/fine-tuning TrackNet), not a threshold you
  can tune away. Be honest about this distinction.

WHAT YOU CAN DO (via tools): read the per-frame feature traces to diagnose a
specific moment, list/score rallies, tune the global clamped parameters,
trigger reprocessing, and save notes. Learning is GLOBAL by default — an
apply_params change becomes the baseline for every future clip; only mark
clip-specific oddities as clip_local (via notes).

WHAT YOU CANNOT DO: edit the CV source code (detector models, tracking, bounce
geometry). Those changes are made by the coding agent in the terminal. When a
fix genuinely needs source edits, DO NOT pretend you changed code — instead
save a precise, self-contained note (frames involved, observed vs expected,
your hypothesis) with add_note so it can be picked up as a handoff, and tell
the user you've done so.

STYLE: Lead with the answer. When the user points at a symptom, pull the
feature trace for those frames before theorizing. Prefer taking the safe
action (tune a param, reprocess, save a handoff note) over describing it, but
say what you changed and why. Keep it concrete."""


@router.post("/chat")
def chat(req: ChatRequest):
    if not ANTHROPIC_API_KEY:
        raise HTTPException(400, "ANTHROPIC_API_KEY not set in .env — the "
                                 "assistant needs it (same key as the report).")
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    system = SYSTEM + "\n\n--- CURRENT STATE ---\n" + _state_snapshot(req.video_id)
    messages = [{"role": m.role, "content": m.text} for m in req.messages
                if m.role in ("user", "assistant")]
    if not messages or messages[-1]["role"] != "user":
        raise HTTPException(400, "last message must be from the user")

    actions: list[dict] = []
    state_changed = False
    try:
        for _ in range(10):  # bounded agentic loop
            resp = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=8000,
                thinking={"type": "adaptive"},
                system=system,
                tools=TOOLS,
                messages=messages,
            )
            if resp.stop_reason != "tool_use":
                break
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for block in resp.content:
                if block.type == "tool_use":
                    out = _dispatch(req.video_id, block.name, block.input)
                    if block.name in ("apply_params", "reprocess", "add_note"):
                        state_changed = True
                    actions.append({"tool": block.name, "input": block.input})
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": out})
            messages.append({"role": "user", "content": results})
    except anthropic.APIStatusError as exc:
        raise HTTPException(502, f"Anthropic API error {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise HTTPException(502, "could not reach the Anthropic API") from exc

    reply = "\n".join(b.text for b in resp.content if b.type == "text").strip()
    return {"reply": reply or "(no text response)", "actions": actions,
            "state_changed": state_changed}
