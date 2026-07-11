# Rally Analysis & Coaching Feedback — Multi-Agent Build Prompts

Snapshot this was designed against: `2026-07-11` codebase context (branch `rally-analysis`, HEAD `d8eb5fc`). **Every prompt below tells the agent to verify real schema/field names first** — this plan was written from documentation, not a live repo read, and your segmentation code is actively changing.

---

## How to run this (one-time, no manual copy-pasting between waves)

1. Save this file at the repo root, e.g. `CV-Training-Tool/rally_analysis_agent_prompts.md`.
2. Open a single Claude Code session in that repo root.
3. Paste this one kickoff message and nothing else:

```
Read rally_analysis_agent_prompts.md in this repo and execute the orchestration plan in
its "Orchestration instructions" section exactly as written. Do the steps marked "you do
this directly" yourself. For steps marked "dispatch as parallel subagents," use your Task
tool with isolation: worktree, launching all agents for that wave in a single message so
they actually run concurrently, each given the exact corresponding numbered prompt from
this file as its full instructions. Wait for a wave's subagents to finish, merge each
worktree branch back per the merge instructions, remove the worktrees, then proceed to
the next wave. Report progress as you go and summarize at the end.
```

That's it — one session, one paste. Claude Code itself manages worktree creation, dispatches the parallel subagents per wave, merges them back, and runs the solo steps (foundation, integration) directly. You don't need to open separate terminals or copy individual prompts yourself.

---

## 0. Hard constraints (apply to every agent, every wave)

- **Never edit** `backend/app/cv/*`, `backend/app/refine/*`, `rally_logic.py`, `pass1.py`, `pass2.py`, or anything your collaborator owns for rally segmentation. Read-only against those.
- **Only read** from `rally_labels`, `overlays`, `ball_corrections`/`ball_tracks`, and court calibration data. Never write to them.
- All new backend code lives under `backend/app/rallyanalysis/`. All new API routes live in `backend/app/api/rally_analysis.py`. All new frontend code lives in a new `frontend/src/RallyAnalysis/` directory plus one tab registration line in `Workspace.jsx`.
- New DB tables only — additive migrations, never alter existing tables.
- v1 scope excludes rally winner/point-outcome, Speedups %, and Finishing Ability (parked until ball tracking improves). Do not build these.
- Contact/shot-event detection must be **pose-based** (swing kinematics off YOLO11-pose tracking), not dependent on `bounce.py` hit detection — ball tracking is currently unreliable, especially far-baseline.

---

## Shared contracts (freeze these in Wave 0, everyone else imports them)

**`ContactEvent`** — one per detected swing/shot:
```
contact_id, video_id, rally_id, frame_idx, timestamp_s,
player_id, team,               # team = whatever serving/receiving or side label the existing serve-formation code uses — verify in players.py
court_x, court_y,              # from existing homography at that frame
shot_side: "left" | "right",
hand_speed_peak: float,
confidence: float 0-1,
is_serve: bool
```

**`ShotFeatureVector`** — one per `ContactEvent`, for coaching:
```
contact_id,
category: "footwork" | "paddle_prep" | "contact_point" | "follow_through",
knee_bend_deg, shoulder_rotation_deg, elbow_extension_deg,
weight_transfer_score, contact_height_m,
follow_through_completion_score,
split_step_delay_s,
deviation_score: float          # signed distance from good-form reference range, per category
```

**`RallyMetrics`** — one per rally:
```
rally_id, video_id,
kitchen_arrival: {player_id: bool},      # reached NVZ while own team serving, this rally
forward_pressure: {player_id: float},    # avg net-ward court depth this rally
shot_side_pct: {player_id: {"left": float, "right": float}},
shot_count: int,
rally_length_bucket: "short" | "medium" | "long",   # 1-5 / 6-10 / 11+ shots
active_duration_s: float
```

**`CoachingReport`** — one per video, generated once:
```
video_id,
category_ratios: {category: {flagged_count, active_rally_seconds, ratio}},
worst_shots: [{contact_id, category, deviation_score, timestamp_s, rally_id, feature_vector}],
best_shots:  [{contact_id, category, timestamp_s, rally_id, feature_vector}],
narrative: str          # single Anthropic call output
```

---

## Orchestration instructions (Claude Code: read and execute this section)

You are the orchestrator for this build. Some steps you do directly in the main session; others you dispatch as parallel Task-tool subagents with `isolation: worktree` so they can't collide on files. Work through the steps in order — do not start a wave until the previous one is merged.

| Step | Who | What |
|---|---|---|
| 1 | **You, directly** | Foundation — run **Prompt 0** yourself in the main session. Commit the result to a new branch `rally-analysis-features`. |
| 2 | **3 parallel subagents** | Wave 1 — dispatch three Task subagents in one message, `isolation: worktree`, each branching from `rally-analysis-features`. Give them **Prompt 1**, **Prompt 2**, **Prompt 3** respectively, one prompt per subagent, verbatim. |
| 3 | **You, directly** | Merge Wave 1 — for each of the three worktree branches, merge it into `rally-analysis-features` (resolve trivial conflicts, run tests after each merge), then remove that worktree. |
| 4 | **2 parallel subagents** | Wave 2 — dispatch two Task subagents in one message, `isolation: worktree`, both branching from the now-updated `rally-analysis-features`. Give them **Prompt 4** and **Prompt 5** respectively. |
| 5 | **You, directly** | Merge Wave 2 — same merge/test/remove-worktree procedure as step 3. |
| 6 | **You, directly (or one subagent if your context is getting long)** | Wave 3 — run **Prompt 6** yourself, or dispatch it as a single `isolation: worktree` subagent if you'd rather preserve your own context window. Merge it back if you used a subagent. |
| 7 | **You, directly** | Integration — run the **Wave 4** prompt yourself against the fully-merged `rally-analysis-features` branch. |
| 8 | **You, directly** | Report back: what was built, what tests pass, any contract mismatches you had to reconcile between subagents, and confirmation that `backend/app/cv/`, `backend/app/refine/`, `rally_logic.py`, `pass1.py`, `pass2.py` were never touched. |

**Why step 1 runs alone:** every later agent imports the same contract definitions and only one edit should ever touch `Workspace.jsx`'s tab list. Skipping straight to parallel work here is exactly how you'd end up with four different subagents fighting over that one file.

**Merge procedure to use at steps 3 and 5** (adapt exact commands to what you find, this is the intent):
```bash
git checkout rally-analysis-features
git merge feat/<branch-name>          # once per finished subagent worktree
# run the repo's test suite; fix trivial merge fallout, not agent logic
git worktree remove ../<worktree-dir>
git branch -d feat/<branch-name>
```

**If your Task tool's `isolation: worktree` doesn't auto-create branches per subagent**, do it manually before dispatching each wave:
```bash
git worktree add ../cvtool-<name> -b feat/<name> rally-analysis-features
```
and instruct each subagent to `cd` into its assigned worktree path as its first action, confirming it's on the right branch before touching any files.

---

## Prompt 0 — Foundation & Contracts (run first, solo)

```
You're working in the CV-Training-Tool pickleball analysis repo. We are adding a new
"Rally Analysis & Coaching" feature set that must NEVER touch backend/app/cv/, 
backend/app/refine/, rally_logic.py, pass1.py, or pass2.py — a collaborator is actively 
changing rally segmentation there. Everything we build only READS from the rally_labels, 
overlays, and calibration tables/data that already exist.

First, read backend/app/db.py, backend/app/cv/players.py, backend/app/cv/court.py, and 
backend/app/cv/overlays.py to confirm the actual schema and field names for: rally_labels 
(columns, meta JSON shape, whether a serving-team field already exists), overlays (how 
per-frame player pose keypoints and track IDs are stored), and court homography helpers 
(how to convert pixel coords to court meters, and where the net/kitchen-line geometry 
constants live). Do not assume the field names below are exact — adapt them to what you find.

Then build:

1. backend/app/rallyanalysis/ package with __init__.py and contracts.py defining these 
   dataclasses (Python dataclasses or pydantic, match whatever style the rest of the 
   backend uses): ContactEvent, ShotFeatureVector, RallyMetrics, CoachingReport. Use the 
   field lists below as a starting point, adjusted for what you found in step 1:
   [paste the four contract blocks from the "Shared contracts" section above]

2. New additive SQLite tables (in db.py, following the existing schema style): 
   contact_events, coaching_reports. Do not alter any existing table.

3. backend/app/api/rally_analysis.py — an empty router with a single 
   GET /api/rally-analysis/health endpoint returning {"status": "ok"}. Register it in 
   main.py the same way other routers are registered.

4. frontend/src/RallyAnalysis/RallyAnalysisPanel.jsx — an empty panel component that 
   renders a placeholder "Rally Analysis" heading and a stub for a clip-list component 
   (props: title, items — each item has {label, timestampS, onSeek}). Also create 
   frontend/src/RallyAnalysis/ClipList.jsx implementing that stub cleanly (styled list, 
   click calls onSeek). Register ONE new tab "Rally Analysis" in Workspace.jsx pointing 
   at RallyAnalysisPanel — this is the only edit anyone should ever make to Workspace.jsx's 
   tab list for this feature, so make it clean.

Commit when done. This branch is the base every other agent will worktree off of.
```

---

## Wave 1 — run these three in parallel worktrees

### Prompt 1 — Contact/Swing Event Detection

```
Repo: CV-Training-Tool. Branch off rally-analysis-features (Wave 0 is already merged there — 
confirm backend/app/rallyanalysis/contracts.py and the contact_events table exist before 
starting; if they don't, stop and flag it rather than redefining them).

Goal: detect shot/contact events per rally using player pose data ONLY — do not use 
bounce.py, ball_track.py, or any ball-trajectory-based hit detection. Ball tracking is 
unreliable (especially far-baseline) and must not be a dependency for this.

Approach: for each rally window (from rally_labels), for each tracked player (from 
overlays' pose/track data), compute per-frame velocity/acceleration of the hitting-side 
wrist/hand keypoint. Detect local peaks that exceed a threshold relative to that player's 
own baseline motion (not a global constant — players' typical motion energy varies). Each 
peak above threshold becomes a ContactEvent: frame_idx, player_id, team, court_x/court_y 
(via existing homography helper), shot_side (left/right relative to court centerline), 
hand_speed_peak, a confidence score (based on how far above threshold), and is_serve 
(true if it's the first detected contact in the rally and matches serve-formation 
positioning from players.py).

Guard against false positives from non-shot arm motion (serve toss, grip adjustment, 
gesturing) — add reasonable filtering (e.g. minimum time gap since last contact for the 
same player, sanity-check against player facing/stance if that data is available).

Write results to the contact_events table via contracts.ContactEvent. Add a script or API 
endpoint (extend rally_analysis.py) to run detection for a given video_id and return the 
events. Write a few unit tests against synthetic pose sequences (clear swing vs. no swing) 
to validate the peak-detection logic in isolation.

Do not build anything downstream (no metrics, no coaching) — just detection and storage.
```

### Prompt 2 — Position-Based Rally Metrics

```
Repo: CV-Training-Tool. Branch off rally-analysis-features (Wave 0 merged). This agent does 
NOT depend on contact/shot detection — it only needs player position per frame plus rally 
and serve-team info, all of which already exist.

First confirm (read players.py, court.py, rally_labels usage) whether serving-team is 
already recorded per rally. If it's not, derive it from the existing serve-formation 
detection in players.py at the start of each rally window, and note in a comment that this 
is a derived heuristic, not ground truth.

Build two metrics in backend/app/rallyanalysis/position_metrics.py:

1. Kitchen Arrival %: for each player, across all rallies where their team was serving, 
   what fraction of those rallies did they have at least one frame inside the 
   non-volley-zone (kitchen) on their side? Use existing court geometry constants (NVZ 
   6.096 x 13.4112m court, kitchen depth) from court.py — do not redefine court dimensions.

2. Forward Pressure: for each player, average net-ward court depth position (distance from 
   their baseline toward the net) across a rally, aggregated per player across all rallies 
   they appear in.

Output both as part of RallyMetrics (contracts.py) per rally, plus a video-level aggregate 
(overall % / average per player). Add an endpoint in rally_analysis.py: 
GET /api/rally-analysis/{video_id}/position-metrics returning per-player aggregates in a 
shape reasonable for a gradient/dominance-bar UI (a value scaled to indicate which side/
player dominates each metric).

Write a couple of tests with synthetic per-frame position data to confirm the aggregation 
math (e.g. a player who's in the kitchen every rally should show 100%).
```

### Prompt 3 — Frontend Tab Build-Out

```
Repo: CV-Training-Tool. Branch off rally-analysis-features (Wave 0 merged — 
RallyAnalysisPanel.jsx and ClipList.jsx already exist as stubs, do not recreate the tab 
registration in Workspace.jsx).

Build the UI inside frontend/src/RallyAnalysis/ against MOCKED data matching these shapes 
(you will swap in real API calls in the integration pass later — for now, hardcode a 
realistic mock JSON object for each and build against it):

1. DominanceBar.jsx — a gradient bar component (blue-to-orange dominance gradient, marker 
   position indicating which side/player dominates a metric) for Kitchen Arrival % and 
   Forward Pressure, similar in spirit to a two-team comparison bar. Props: label, 
   leftLabel, rightLabel, leftValue, rightValue (or a single -1..1 dominance score).

2. StatBlock.jsx — simple percentage callout blocks (big number + label + player 
   thumbnail/initial), for things like per-player kitchen arrival %.

3. Wire RallyAnalysisPanel.jsx to lay out: a DominanceBar section (Forward Pressure / 
   Kitchen Arrival), a shot-distribution section (reuse StatBlock or a stacked-bar variant 
   for left/right % per player — stub this even though the data comes in Wave 2), and the 
   ClipList component (from Wave 0) rendered twice — one instance titled "Highlights" and 
   one titled "Coaching Moments" — both empty/mocked for now, populated later.

Use mock data files (frontend/src/RallyAnalysis/mockData.js) matching the RallyMetrics and 
CoachingReport shapes from contracts.py (mirror the field names) so swapping to real API 
calls later is a near-zero diff.

Keep this component tree self-contained — nothing here should require backend changes.
```

---

## Wave 2 — run these two in parallel worktrees (after Wave 1 merges)

### Prompt 4 — Shot-Based Metrics

```
Repo: CV-Training-Tool. Branch off rally-analysis-features with Wave 1 merged in — confirm 
the contact_events table is populated (or at least that ContactEvent detection code exists 
and runs) before starting.

Build backend/app/rallyanalysis/shot_metrics.py consuming ContactEvent rows:

1. Shot Side Distribution: per player/team, % of contacts on left vs right side of court 
   (from ContactEvent.shot_side).

2. Rally Length buckets: per rally, shot_count = count of ContactEvents in that rally's 
   frame range; bucket into short (1-5), medium (6-10), long (11+). Make the thresholds 
   named constants, not magic numbers, so they're easy to tune later.

Populate these into RallyMetrics (contracts.py) alongside the position metrics from Wave 1 
— check position_metrics.py's output shape and match it rather than inventing a parallel 
structure. Add/extend an endpoint: GET /api/rally-analysis/{video_id}/shot-metrics.

Add tests with synthetic ContactEvent lists to confirm bucket boundaries and side-% math.
```

### Prompt 5 — Coaching Feedback Pipeline

```
Repo: CV-Training-Tool. Branch off rally-analysis-features with Wave 1 merged in — confirm 
ContactEvent detection exists before starting; this entire pipeline is triggered off 
contact events, not ball/bounce data.

Build backend/app/rallyanalysis/coaching.py:

1. For each ContactEvent, extract a ShotFeatureVector (contracts.py) using pose keypoints 
   at and around the contact frame (a short window for follow-through/split-step, not just 
   the single contact frame): knee_bend_deg, shoulder_rotation_deg, elbow_extension_deg, 
   weight_transfer_score, contact_height_m (relative to net height via court.py), 
   follow_through_completion_score, split_step_delay_s. Assign a category (footwork / 
   paddle_prep / contact_point / follow_through) and compute a deviation_score against a 
   reasonable good-form reference range per category (hardcode sensible biomechanical 
   ranges as named constants — flag clearly in comments that these are starting-point 
   heuristics a human coach should sanity check, not validated ground truth).

2. Aggregate ratios per category: flagged_count / active_rally_seconds (sum of rally 
   durations for the video, NOT total video length — exclude dead time between rallies). 
   Use rally_labels start/end to compute active duration.

3. Select worst_shots and best_shots using an ADAPTIVE severity threshold, not a fixed 
   top-N — e.g. anything beyond N standard deviations from the category's reference range 
   (worst) or clearly within-range with high confidence (best). A clean session might 
   surface 1 shot, a rough one might surface 8+; don't hardcode a count.

4. One Anthropic API call per video (reuse whatever client setup 
   backend/app/analysis/reporter.py already uses for the discrepancy reporter — same 
   pattern, same env vars, don't build a second client config). Input: category_ratios, 
   worst_shots (with feature vectors + timestamps + rally context), best_shots. Prompt it 
   to give specific, actionable advice per flagged category and explicitly reinforce what's 
   working from the best shots. Store the result in the coaching_reports table via 
   CoachingReport.

Add endpoint: POST /api/rally-analysis/{video_id}/coaching-report (triggers generation) 
and GET .../coaching-report (returns cached result, regenerate only on request — this 
should not re-run on every page load, it's an LLM call).

Test the ratio math and adaptive-threshold selection with synthetic feature vectors; you 
don't need to test actual LLM output quality, just that the right data reaches the prompt.
```

---

## Wave 3 — single agent (after Wave 2 merges)

### Prompt 6 — Highlight Tagging

```
Repo: CV-Training-Tool. Branch off rally-analysis-features with Wave 2 merged in — confirm 
shot_metrics.py (shot_count, rally_length_bucket) and position_metrics.py (player 
positions per rally) both exist before starting.

Build backend/app/rallyanalysis/highlights.py that classifies each rally into zero or more 
tags using rule-based logic over existing RallyMetrics — no new detection needed:

- "Long Rally": rally_length_bucket == "long"
- "Firefight": high shot rate (shots per second above a threshold) with both players' 
  forward_pressure indicating they were both near the net for a sustained stretch
- "Poach": a player's court_x crosses into their partner's side during the rally (requires 
  per-player court_x over time within the rally — pull from the same position data Wave 1 
  used, don't refetch differently)

Make every threshold a named constant. Output a list of {rally_id, tag, timestamp_s 
(rally start), video_id} rows. Add endpoint: 
GET /api/rally-analysis/{video_id}/highlights.

Wire this into the frontend "Highlights" ClipList instance (from Wave 1's 
RallyAnalysisPanel.jsx) — replace its mock data with a real fetch to this endpoint, using 
ClipList's existing onSeek prop to jump the video player to timestamp_s. Also wire the 
Coaching Feedback Pipeline's worst_shots/best_shots (Wave 2, coaching-report endpoint) into 
the "Coaching Moments" ClipList instance the same way.

This prompt does both the backend classification AND the two frontend data-wiring edits — 
keep them together since they're the same conceptual change (mock -> real data) touching 
the same two ClipList call sites.
```

---

## Wave 4 — Integration pass (solo, final)

```
Repo: CV-Training-Tool, rally-analysis-features branch with all prior waves merged.

Do a full pass to catch integration drift between agents that worked in parallel:

1. Diff contracts.py usage across position_metrics.py, shot_metrics.py, coaching.py, and 
   highlights.py — confirm they actually share the same RallyMetrics/ContactEvent shape 
   rather than each having drifted into slightly different field names. Reconcile any 
   mismatches.

2. Consolidate rally_analysis.py into one coherent router with a clear set of endpoints 
   (list them out in a comment block at the top of the file) rather than four separately-
   added endpoint groups bolted together.

3. Confirm RallyAnalysisPanel.jsx now pulls all real data (position metrics, shot metrics, 
   coaching report) — no leftover mockData.js imports for anything that has a real 
   endpoint now. Coaching report generation should be an explicit user action (a button), 
   not an automatic fetch, since it's an LLM call.

4. Run the existing test suite plus the new tests added across all waves; fix any breakage 
   from integration, not from re-litigating each agent's individual logic.

5. Confirm nothing in backend/app/cv/, backend/app/refine/, rally_logic.py, pass1.py, or 
   pass2.py was touched anywhere across the whole feature branch (git diff against main 
   for those paths should be empty).

Once this passes, this branch is ready to rebase onto whatever your collaborator's latest 
segmentation branch is, and the merge should be close to conflict-free given the read-only 
contract this feature has held to throughout.
```
