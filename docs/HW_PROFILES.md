# Hardware profiles (`HW_PROFILE`)

A per-machine bundle of **speed** knobs — device, precision, OpenVINO on/off,
TrackNet on/off, Pass 1 subsampling. Point is to tune each person's GPU for
their own throughput. **Not** for making two machines produce identical
output — that is explicitly not a goal.

Separate from `LearnedParams` (`backend/app/learning/params.py`): those are
detection-quality tunables that compound globally via the report. Hardware
profiles never touch that table.

## Quick start

In `.env`:
```
HW_PROFILE=cuda
```
Unset (or a name not in `hw_profiles.json`... unset is fine, an unknown name
errors — see below) → today's plain `CV_DEVICE`/`TRACKNET_ENABLED`/`PASS2_BALL`
behavior, unchanged.

## Where it lives

- **`backend/app/hw_profile.py`** — the schema (`PROFILE_DEFAULTS`) and the
  loader (`active_profile()`). This is the only file that knows the shape of
  a profile.
- **`backend/app/hw_profiles.json`** — the actual profiles, committed to git
  (unlike `data/`, which is gitignored) so a branch carries everyone's
  profile. **Adding a machine = adding a JSON block here. No code change.**

Current profiles:

| Name | Machine | device | precision | openvino | tracknet | pass2_ball |
|---|---|---|---|---|---|---|
| `cuda` | this machine (NVIDIA) | `cuda` | `fp16` | off | on | on |
| `arc140v` | partner's machine (Intel Arc 140V) | `xpu` | `fp16` | on | on | off |

**Note (2026-07-21):** the `cuda` row above has said `tracknet: on` since this
file was written, but `HW_PROFILE` is not actually set in the EC2 worker's
deployed env (`scripts/restart-ec2.sh`) — so this row was never the thing
actually turning TrackNet on there; `TRACKNET_ENABLED` stayed at its
env-unset default of `False` (see Precedence rules below) until fine-tuned
weights existed to make enabling it worthwhile. `restart-ec2.sh` now sets
`TRACKNET_ENABLED=1` directly instead of adopting this profile — simpler
given the script already sets `CV_DEVICE`/`PASS2_BALL` the same explicit way,
and `HW_PROFILE=cuda` would also silently pull in `precision`/`openvino`
overrides this deployment hasn't evaluated. See `PASS1_PASS2_SPEC.md`'s
"Arc-fit noise filter" / ball-observation section for why this now matters
and the new S3 sync path for the weights file
(`storage_s3.upload_tracknet_weights` / `cloud_ingest._sync_tracknet_weights`,
mirroring how calibration syncs — see those two functions' docstrings, the
closest thing to written documentation of that mechanism right now; a
top-level AWS/deploy doc describing the hybrid S3 architecture doesn't
currently exist in this repo despite being referenced elsewhere).

## Adding a new profile

Edit `backend/app/hw_profiles.json`, add a block:
```json
"my-gpu-name": {
  "device": "cuda",        // xpu | cuda | cpu
  "precision": "fp16",     // fp16 | fp32
  "openvino": false,       // Intel-only; leave off on non-Intel GPUs
  "tracknet": true,
  "pass2_ball": true
}
```
Then set `HW_PROFILE=my-gpu-name` in that machine's own `.env`. That's it —
nothing else to touch.

Optional keys (omit unless you actually need them):
```json
  "p1_scale": 0.3,        // overrides LearnedParams for THIS machine only
  "p1_frame_step": 4      // (see "p1_scale / p1_frame_step" below)
```

## Every knob and what it plugs into

| Profile key | Consumer | Behavior |
|---|---|---|
| `device` | `cv/device.py` `torch_device()` | Checked before `CV_DEVICE`; `None`/no profile → old `CV_DEVICE` path. |
| `precision` | `cv/players.py` (pose OpenVINO export + torch `.track(half=...)`), `cv/ball_tracknet.py` (`.model.half()`) | `"fp16"` (default) matches today's behavior (pose OV export was always `half=True`); `"fp32"` opts out. Skipped on CPU regardless (fp16 on CPU is unsupported/slow). |
| `openvino` | `cv/players.py` `_openvino_pose_dir`, `cv/ball_tracknet.py` `_openvino_infer` | `False` short-circuits the OpenVINO export/compile attempt entirely — set this on non-Intel GPUs to avoid a wasted export. |
| `tracknet` | `config.py` → `TRACKNET_ENABLED` | Only used if the `TRACKNET_ENABLED` env var is **unset**; an explicit env var always wins. |
| `pass2_ball` | `config.py` → `PASS2_BALL_ENABLED` | Same precedence as `tracknet`. |
| `p1_scale`, `p1_frame_step` | `pipeline.py` `_with_hw_overrides()` | See below — the one tricky case. |
| `pose_batch` | *(nothing yet)* | Schema placeholder. Pose inference is frame-serial today (`PoseTracker.update()` calls `model.track()` one frame at a time); there's no batch path to plug a batch size into without restructuring the Pass 2 loop. Add real wiring only once someone measures a GPU-bound pose stage. |

## `p1_scale` / `p1_frame_step` — the one subtlety

These two names already exist as **LearnedParams** — versioned, DB-stored,
tuned globally by the discrepancy report. A hardware profile is NOT allowed
to write to that table (would corrupt the "global learning" story for the
other machine).

So instead: `pipeline.py`'s `_with_hw_overrides()` takes the LearnedParams
dict `get_latest()` just fetched and, **only if the active profile sets
these keys**, overlays them on an in-memory copy before Pass 1/Pass 2 run.
The DB row is never touched. Leave them unset (the default) to use whatever
the learned baseline says.

Practical effect: a weak GPU can set a smaller `p1_scale` / bigger
`p1_frame_step` so Pass 1 leans harder on subsampling and Pass 2 runs
lighter — without that override ever polluting the shared learned baseline
the other machine (or the report) sees.

## Precedence rules

- **Explicit env var always beats the profile.** `TRACKNET_ENABLED=0` in
  `.env` wins even if `HW_PROFILE`'s `tracknet` is `true`.
- **Unset `HW_PROFILE` → zero behavior change.** Every consumer falls back to
  exactly what it did before this feature existed.
- **Unknown `HW_PROFILE` name → hard error** (`SystemExit` listing the valid
  names) rather than silently falling back — a typo'd profile name should
  never quietly run on the wrong settings.

## Known interactions

- **autolearn / model_registry** don't reference device/precision at all —
  safe to run under any profile.
- **Learned values were measured under specific hardware** (e.g.
  `pose_imgsz=1920` benchmarks, ball thresholds) — switching device/precision
  can shift detection results slightly. Expected and fine given the explicit
  "no cross-machine reproducibility" goal; don't "fix" this later.
- **Device/precision resolve once per process** (`lru_cache`d in
  `device.py`, module-level in `config.py`) — changing `HW_PROFILE` needs a
  **backend restart**. This is also why there's no frontend toggle: a
  Setup-tab dropdown would imply live switching that doesn't actually work.

## Verifying

```powershell
cd backend
./.venv/Scripts/python.exe -c "from app.hw_profile import active_profile; print(active_profile())"
```
`None` with `HW_PROFILE` unset; the resolved dict otherwise.
