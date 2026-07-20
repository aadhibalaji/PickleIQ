# Connecting an external ball tracker (e.g. MTI + temporal integration)

Your tracker runs independently (any language/framework, offline is fine) and
produces a **per-frame ball track**. You POST that track to the app; **Pass 2
prefers it over its built-in classical detector**. Nothing else in the pipeline
needs to change — better ball positions flow straight into height
reconstruction, bounce/terminal-event detection, net-crossing logic, serve/pass
rejection, and boundary accuracy.

## Ball-source priority in Pass 2
`manual correction` (per-frame) > **`your external track`** > `TrackNet`
(off by default) > `classical MotionBallDetector` (fallback).
Frames you don't provide fall back to the classical detector automatically, so a
partial track (e.g. only the frames your tracker is confident on) is fine.

## Endpoint
```
POST /api/videos/{video_id}/ball_track
Content-Type: application/json

{
  "source": "mti_v1",                       // free label for which tracker produced it
  "track": [
    { "frame": 401, "x": 962.4, "y": 540.1, "diameter": 7.8, "confidence": 0.94 },
    { "frame": 402, "x": 968.0, "y": 536.7, "diameter": 7.5, "confidence": 0.91 },
    ...
  ]
}
```
- `GET  /api/videos/{video_id}/ball_track` → `{ source, n_frames, created_at }`
- `DELETE /api/videos/{video_id}/ball_track` → remove it

Get the video id / dimensions / fps / frame_count from `GET /api/videos` and
`GET /api/videos/{id}`.

## Coordinate + frame conventions (important)
- **`frame`**: 0-based frame index of the video, decoded in order (the same
  indexing Pass 2 uses). Decode the exact stored file (stream it from
  `GET /api/videos/{id}/stream`) so your frame numbers line up.
- **`x`, `y`**: ball centre in **full-resolution image pixels**, origin top-left
  (i.e. native `width`×`height` of the video — NOT a resized/letterboxed frame).
- **`diameter`** (optional, but **matters**): the ball's apparent diameter in px.
  This is what drives the size→depth **height reconstruction**, which is what
  makes **bounce / rally-END detection** work. Without it, height is unknown and
  end-detection stays on the coarse fallback — but good `x,y` alone still
  materially improves **serve-start detection, net-crossing counting, and the
  serve-vs-pass rejection** (all of which only need position). Provide it if your
  tracker can estimate the blob size.
- **`confidence`** (optional, default 0.9): 0–1. Currently informational; low-
  confidence frames are still used if present (omit a frame instead to fall back
  to the classical detector there).

## After uploading
Reprocess Pass 2 so it consumes the track: **Report tab → "Regenerate video
(Pass 2)"**, or `POST /api/videos/{id}/reprocess?which=pass2`. The server logs
`pass2 <id>: using external ball track (N frames)` when it picks it up.

## Minimal Python example
```python
import requests, json
BASE = "http://127.0.0.1:8420/api"
video_id = requests.get(f"{BASE}/videos").json()[0]["id"]

# your tracker -> list of {frame, x, y, diameter?, confidence?}
track = run_my_mti_tracker(video_id)           # you implement this

requests.post(f"{BASE}/videos/{video_id}/ball_track",
              json={"source": "mti_v1", "track": track}).raise_for_status()
requests.post(f"{BASE}/videos/{video_id}/reprocess?which=pass2")
```

## Why this helps (ties the whole system together)
Ball tracking is the current accuracy ceiling: the rally-END model and the
serve-vs-pass logic are both capped by noisy far-side ball positions. A clean MTI
track removes that ceiling — reliable net-crossings make pass-rejection exact,
and (with `diameter`) reliable bounces make the terminal-event/end detection fire
instead of falling back to the window edge. It's also the efficient path: no
per-frame heatmap CNN in the hot loop, just the positions you already computed.

## Tighter (in-process) alternative
If you'd rather have your model called *live* inside Pass 2 (instead of
precompute + upload), it can implement the same tiny interface as
`backend/app/cv/ball_tracknet.py::TrackNetBall` (an `available` flag +
`infer_triplet(frames)->(x,y,diameter,confidence)|None`) and be slotted into the
priority chain in `backend/app/cv/pass2.py`. The upload path above is
recommended — it decouples your tracker from the CV process and is language-agnostic.
