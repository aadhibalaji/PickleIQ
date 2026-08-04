"""S3-backed video storage for the STORAGE_BACKEND=s3 deployment path.

Upload: local bytes -> S3 (key structure {user_id}/{video_id}.mp4). Playback:
a presigned GET URL, so the browser streams directly from S3 (range requests
included for free) — see api/media.py stream_video. CV processing
(cv/pass1.py, cv/pass2.py) still needs a real local file — neither reads S3 —
so download_to_scratch() pulls the object back down to
DATA_DIR/scratch/{video_id}.mp4 before a pass runs, and cleanup_scratch()
removes it once the pass1->pass2->contacts chain finishes (see pipeline.py).

Uses the instance's IAM role automatically via boto3 (same pattern as
secrets_bootstrap.py / self_stop.py) — never pass explicit credentials.

Unlike secrets_bootstrap.py/self_stop.py, failures here are NOT swallowed:
an upload/download the caller believes succeeded but silently didn't is
worse than a loud 500 — a misconfigured bucket or missing IAM permission
should fail clearly, not disappear into a log line.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from .config import DATA_DIR, S3_VIDEO_BUCKET

log = logging.getLogger("storage_s3")

_REGION = os.environ.get("AWS_REGION", "us-east-2")
DEFAULT_USER_ID = "default"   # placeholder until real multi-tenancy exists


def _client():
    import os
    import boto3
    profile = os.environ.get("AWS_PROFILE_OVERRIDE")
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    return session.client("s3", region_name=_REGION)


def _require_bucket() -> None:
    if not S3_VIDEO_BUCKET:
        raise RuntimeError("STORAGE_BACKEND=s3 but S3_VIDEO_BUCKET is not set")


def video_key(video_id: str, user_id: str = DEFAULT_USER_ID) -> str:
    return f"{user_id}/{video_id}.mp4"


def list_video_keys(user_id: str = DEFAULT_USER_ID) -> list[str]:
    """Every uploaded-video object key currently under {user_id}/ (see
    video_key) — for the EC2 worker's cloud_ingest poller to diff against
    its own DB and discover videos it hasn't started processing yet. Scoped
    to this prefix specifically so it never picks up results/*.json
    objects, which live under a separate top-level prefix."""
    _require_bucket()
    prefix = f"{user_id}/"
    try:
        resp = _client().list_objects_v2(Bucket=S3_VIDEO_BUCKET, Prefix=prefix)
    except Exception as exc:
        log.error("S3 list_objects_v2 failed: bucket=%s prefix=%s (%s)", S3_VIDEO_BUCKET, prefix, exc)
        raise RuntimeError(f"S3 list failed for prefix {prefix!r} in bucket {S3_VIDEO_BUCKET!r}: {exc}") from exc
    return [obj["Key"] for obj in resp.get("Contents", [])]


def upload_video(local_path: Path, video_id: str, user_id: str = DEFAULT_USER_ID) -> str:
    """Uploads local_path to S3, returns the object key."""
    _require_bucket()
    key = video_key(video_id, user_id)
    try:
        _client().upload_file(str(local_path), S3_VIDEO_BUCKET, key)
    except Exception as exc:
        log.error("S3 upload failed: bucket=%s key=%s (%s)", S3_VIDEO_BUCKET, key, exc)
        raise RuntimeError(
            f"S3 upload failed for {key!r} in bucket {S3_VIDEO_BUCKET!r}: {exc}") from exc
    log.info("uploaded %s to s3://%s/%s", local_path, S3_VIDEO_BUCKET, key)
    return key


def generate_presigned_url(s3_key: str, expires_in: int = 3600) -> str:
    _require_bucket()
    try:
        url = _client().generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_VIDEO_BUCKET, "Key": s3_key},
            ExpiresIn=expires_in,
        )
    except Exception as exc:
        log.error("presigned URL generation failed: bucket=%s key=%s (%s)",
                  S3_VIDEO_BUCKET, s3_key, exc)
        raise RuntimeError(f"could not generate presigned URL for {s3_key!r}: {exc}") from exc
    log.info("generated presigned URL for s3://%s/%s (expires in %ds)",
              S3_VIDEO_BUCKET, s3_key, expires_in)
    return url


def scratch_dir() -> Path:
    d = DATA_DIR / "scratch"
    d.mkdir(parents=True, exist_ok=True)
    return d


def scratch_path(video_id: str) -> Path:
    return scratch_dir() / f"{video_id}.mp4"


def download_to_scratch(s3_key: str, video_id: str) -> Path:
    """Returns a local path holding this video's bytes, downloading from S3
    only if a scratch copy isn't already there (fresh upload, or a same-size
    leftover from a prior run on this instance)."""
    _require_bucket()
    dest = scratch_path(video_id)
    try:
        head = _client().head_object(Bucket=S3_VIDEO_BUCKET, Key=s3_key)
        remote_size = head["ContentLength"]
    except Exception as exc:
        log.error("S3 head_object failed: bucket=%s key=%s (%s)", S3_VIDEO_BUCKET, s3_key, exc)
        raise RuntimeError(f"could not read {s3_key!r} from bucket {S3_VIDEO_BUCKET!r}: {exc}") from exc

    if dest.exists() and dest.stat().st_size == remote_size:
        log.info("scratch copy already present for %s (%d bytes) — skipping download",
                  video_id, remote_size)
        return dest

    log.info("downloading s3://%s/%s to %s", S3_VIDEO_BUCKET, s3_key, dest)
    try:
        _client().download_file(S3_VIDEO_BUCKET, s3_key, str(dest))
    except Exception as exc:
        log.error("S3 download failed: bucket=%s key=%s (%s)", S3_VIDEO_BUCKET, s3_key, exc)
        raise RuntimeError(f"S3 download failed for {s3_key!r}: {exc}") from exc
    log.info("download complete: %s (%d bytes)", dest, dest.stat().st_size)
    return dest


def _put_json(key: str, data: dict) -> None:
    """Shared upload body for both the full and per-window results objects —
    same failure discipline as everything else here: log, then raise loud."""
    _require_bucket()
    body = json.dumps(data).encode("utf-8")
    try:
        _client().put_object(Bucket=S3_VIDEO_BUCKET, Key=key, Body=body,
                              ContentType="application/json")
    except Exception as exc:
        log.error("S3 JSON upload failed: bucket=%s key=%s (%s)", S3_VIDEO_BUCKET, key, exc)
        raise RuntimeError(
            f"S3 JSON upload failed for {key!r} in bucket {S3_VIDEO_BUCKET!r}: {exc}") from exc
    log.info("uploaded s3://%s/%s (%d bytes)", S3_VIDEO_BUCKET, key, len(body))


def _get_json(key: str) -> dict | None:
    """None on a missing key (expected — not yet produced), not an error."""
    _require_bucket()
    client = _client()
    try:
        obj = client.get_object(Bucket=S3_VIDEO_BUCKET, Key=key)
        body = obj["Body"].read()
    except client.exceptions.NoSuchKey:
        log.info("no object at s3://%s/%s", S3_VIDEO_BUCKET, key)
        return None
    except Exception as exc:
        log.error("S3 JSON download failed: bucket=%s key=%s (%s)", S3_VIDEO_BUCKET, key, exc)
        raise RuntimeError(
            f"S3 JSON download failed for {key!r} in bucket {S3_VIDEO_BUCKET!r}: {exc}") from exc
    log.info("downloaded s3://%s/%s (%d bytes)", S3_VIDEO_BUCKET, key, len(body))
    return json.loads(body)


def results_key(video_id: str) -> str:
    return f"results/{video_id}/result.json"


def window_results_key(video_id: str, window_index: int) -> str:
    return f"results/{video_id}/window_{window_index}.json"


def progress_key(video_id: str) -> str:
    return f"results/{video_id}/status.json"


def results_exist(video_id: str) -> bool:
    """True if a final result.json already exists for video_id. Used by
    cloud_ingest to avoid re-processing a video whose chain already ran to
    completion in a PRIOR boot of this same instance — protects against the
    dedup check in cloud_ingest._known_video_ids() going stale if this
    instance's own local DB doesn't survive a stop/start cycle (its dedup
    otherwise relies entirely on that DB)."""
    _require_bucket()
    key = results_key(video_id)
    try:
        _client().head_object(Bucket=S3_VIDEO_BUCKET, Key=key)
        return True
    except Exception as exc:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey"):
            return False
        log.error("S3 head_object failed checking for existing results: bucket=%s key=%s (%s)",
                  S3_VIDEO_BUCKET, key, exc)
        raise RuntimeError(f"could not check for existing results at {key!r}: {exc}") from exc


def upload_results(video_id: str, data: dict) -> str:
    """Uploads a video's exported pipeline results (see results_export.py) to
    S3 as JSON. Called from the cloud side once processing finishes, before
    the instance stops — see pipeline._finish_chain."""
    key = results_key(video_id)
    _put_json(key, data)
    return key


def download_results(video_id: str) -> dict | None:
    """Returns the parsed result.json for video_id, or None if the cloud
    hasn't produced one (yet) — a normal, expected state for a video still
    processing or one that was never sent to the cloud, not an error."""
    return _get_json(results_key(video_id))


def upload_window_results(video_id: str, window_index: int, data: dict) -> str:
    """Same as upload_results but for one Pass 2 window's incremental
    overlay snapshot (see results_export.export_window_results). Called from
    inside run_pass2 as each window finishes — well before the chain-level
    final export, so a caller must not let a failure here interrupt Pass 2
    itself (see the non-fatal wrapping at the call site in cv/pass2.py)."""
    key = window_results_key(video_id, window_index)
    _put_json(key, data)
    return key


def download_window_results(video_id: str, window_index: int) -> dict | None:
    return _get_json(window_results_key(video_id, window_index))


def upload_progress(video_id: str, data: dict) -> str:
    """Live single-object progress snapshot for the currently-running Pass 1
    stage (Pass 2 already has an implicit progress signal via its per-window
    objects — see upload_window_results — so this exists specifically to
    fill the Pass 1 gap: Pass 1 processes the whole clip in one pass with no
    natural "window" checkpoints, so instead of many small objects like Pass
    2, it's a single object OVERWRITTEN in place each time — see
    cv/pass1.py's periodic call, and api/media.py's poll-cloud-windows,
    which downloads and mirrors it. Deleted via delete_progress once Pass 1
    finishes, so a later poll can never re-apply a stale 'running' snapshot
    after the video has moved on (see delete_progress's docstring)."""
    key = progress_key(video_id)
    _put_json(key, data)
    return key


def download_progress(video_id: str) -> dict | None:
    return _get_json(progress_key(video_id))


def delete_progress(video_id: str) -> None:
    """Removes the live progress snapshot once Pass 1 finishes. Without
    this, a poll tick arriving after Pass 1 (and possibly the whole chain)
    is already done could still find the last 'running' snapshot written
    before completion and re-mirror it — flipping an already-'done' local
    job status on a UI-only container back to 'running'. Best-effort: a
    failure here must not fail Pass 1 itself, it just risks that one stale
    object lingering (harmless — it'll be overwritten by the next run, or
    ignored once the chain-end result.json supersedes it via sync)."""
    _require_bucket()
    key = progress_key(video_id)
    try:
        _client().delete_object(Bucket=S3_VIDEO_BUCKET, Key=key)
    except Exception as exc:
        log.warning("could not delete progress snapshot for %s (%s)", video_id, exc)


def list_window_objects(video_id: str) -> list[dict]:
    """[{'index': n, 'etag': ...}] for every window_N.json currently in S3
    for this video, sorted by index. The etag is what lets a poller (see
    api/media.py's poll-cloud-windows) tell "new window" apart from "same
    index, reprocessed with different content" — a reprocessed run's Pass 2
    reuses the same window_N.json keys, so index alone can't be trusted as a
    once-ever-seen watermark."""
    _require_bucket()
    prefix = f"results/{video_id}/window_"
    try:
        resp = _client().list_objects_v2(Bucket=S3_VIDEO_BUCKET, Prefix=prefix)
    except Exception as exc:
        log.error("S3 list_objects_v2 failed: bucket=%s prefix=%s (%s)", S3_VIDEO_BUCKET, prefix, exc)
        raise RuntimeError(f"S3 list failed for prefix {prefix!r} in bucket {S3_VIDEO_BUCKET!r}: {exc}") from exc
    out = []
    for obj in resp.get("Contents", []):
        name = obj["Key"].rsplit("/", 1)[-1]
        if name.startswith("window_") and name.endswith(".json"):
            try:
                idx = int(name[len("window_"):-len(".json")])
            except ValueError:
                continue
            out.append({"index": idx, "etag": obj["ETag"]})
    return sorted(out, key=lambda o: o["index"])


def calibration_key() -> str:
    return "calibration/global.json"


def upload_calibration(data: dict) -> str:
    """Uploads the current global CameraCalibration to S3 so the EC2 worker's
    cloud_ingest poller can pick it up and mirror it into its own local DB
    (see cloud_ingest._sync_calibration) — needed because calibration is
    normally saved once through a RUN_PROCESSING=0 UI-only container, whose
    local calibrations table the separate worker instance never otherwise
    sees. Called from api/annotate.py's save_calibration."""
    key = calibration_key()
    _put_json(key, data)
    return key


def download_calibration() -> dict | None:
    """None if no global calibration has ever been saved on any S3-backed
    container yet — expected, not an error."""
    return _get_json(calibration_key())


def self_stop_key() -> str:
    return "self_stop/state.json"


def upload_self_stop_state(enabled: bool) -> str:
    """Mirrors the self-stop toggle to S3 so the EC2 worker's cloud_ingest
    poller can pick it up (see cloud_ingest._sync_self_stop) — same reasoning
    as upload_calibration/upload_refine_config: normally flipped through a
    RUN_PROCESSING=0 UI-only container, whose local state the separate EC2
    worker never otherwise sees. Called from api/media.py's set_self_stop."""
    key = self_stop_key()
    _put_json(key, {"enabled": bool(enabled)})
    return key


def download_self_stop_state() -> dict | None:
    """None if the toggle has never been set from any S3-backed container —
    expected, not an error; caller falls back to the local/default value."""
    return _get_json(self_stop_key())


def tracknet_weights_key() -> str:
    return "models/tracknet_v2.pt"


def upload_tracknet_weights(local_path: Path) -> str:
    """Uploads a (presumably fine-tuned) TrackNet checkpoint to S3 so the EC2
    worker's cloud_ingest poller can pick it up (see
    cloud_ingest._sync_tracknet_weights) — mirrors upload_calibration's
    reasoning: the weights are normally produced/placed on a dev machine, not
    through the EC2 worker itself, so without this it would never see them."""
    _require_bucket()
    key = tracknet_weights_key()
    try:
        _client().upload_file(str(local_path), S3_VIDEO_BUCKET, key)
    except Exception as exc:
        log.error("S3 upload failed: bucket=%s key=%s (%s)", S3_VIDEO_BUCKET, key, exc)
        raise RuntimeError(
            f"S3 upload failed for {key!r} in bucket {S3_VIDEO_BUCKET!r}: {exc}") from exc
    log.info("uploaded %s to s3://%s/%s", local_path, S3_VIDEO_BUCKET, key)
    return key


def download_tracknet_weights(dest: Path) -> bool:
    """Downloads the S3 TrackNet checkpoint to `dest` if it's missing or a
    different size than what's already there. Returns True if a download
    happened, False if dest was already current or nothing exists in S3 yet
    (both non-error, expected states — see cloud_ingest._sync_tracknet_weights)."""
    _require_bucket()
    key = tracknet_weights_key()
    client = _client()
    try:
        head = client.head_object(Bucket=S3_VIDEO_BUCKET, Key=key)
    except Exception as exc:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey"):
            return False
        log.error("S3 head_object failed: bucket=%s key=%s (%s)", S3_VIDEO_BUCKET, key, exc)
        raise RuntimeError(f"could not check {key!r} in bucket {S3_VIDEO_BUCKET!r}: {exc}") from exc

    if dest.exists() and dest.stat().st_size == head["ContentLength"]:
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading s3://%s/%s to %s", S3_VIDEO_BUCKET, key, dest)
    try:
        client.download_file(S3_VIDEO_BUCKET, key, str(dest))
    except Exception as exc:
        log.error("S3 download failed: bucket=%s key=%s (%s)", S3_VIDEO_BUCKET, key, exc)
        raise RuntimeError(f"S3 download failed for {key!r}: {exc}") from exc
    log.info("download complete: %s (%d bytes)", dest, dest.stat().st_size)
    return True


def refine_config_key() -> str:
    return "refine/config.override.yaml.json"


def upload_refine_config(data: dict) -> str:
    """Mirrors the current refine config.override.yaml (serve_gate_min etc.)
    to S3 so the EC2 worker's cloud_ingest poller can pick it up (see
    cloud_ingest._sync_refine_config) — same gap as calibration/TrackNet
    weights: this is normally written through a RUN_PROCESSING=0 UI-only
    container (api/annotate.py's write_refine_override), whose local
    data/refine/config.override.yaml the separate EC2 worker never otherwise
    sees. Called on every POST /refine/config/override."""
    key = refine_config_key()
    _put_json(key, data)
    return key


def download_refine_config() -> dict | None:
    """None if no override has ever been applied on any S3-backed container
    yet — expected, not an error."""
    return _get_json(refine_config_key())


def delete_video_data(video_id: str) -> None:
    """Removes everything this video wrote to S3: the uploaded video object
    and every results/{video_id}/* object (result.json, window_N.json,
    status.json). Best-effort per key — a missing object is not an error,
    but a client/network failure is (same discipline as the rest of this
    module: loud failures, not silent no-ops)."""
    _require_bucket()
    client = _client()
    keys = [video_key(video_id)]
    try:
        resp = client.list_objects_v2(Bucket=S3_VIDEO_BUCKET, Prefix=f"results/{video_id}/")
    except Exception as exc:
        log.error("S3 list_objects_v2 failed: bucket=%s prefix=results/%s/ (%s)",
                  S3_VIDEO_BUCKET, video_id, exc)
        raise RuntimeError(f"S3 list failed for results/{video_id}/: {exc}") from exc
    keys += [obj["Key"] for obj in resp.get("Contents", [])]
    try:
        client.delete_objects(
            Bucket=S3_VIDEO_BUCKET,
            Delete={"Objects": [{"Key": k} for k in keys]},
        )
    except Exception as exc:
        log.error("S3 delete_objects failed: bucket=%s keys=%s (%s)", S3_VIDEO_BUCKET, keys, exc)
        raise RuntimeError(f"S3 delete failed for {video_id!r}: {exc}") from exc
    log.info("deleted %d S3 object(s) for video %s", len(keys), video_id)


def cleanup_scratch(video_id: str) -> None:
    dest = scratch_path(video_id)
    try:
        if dest.exists():
            dest.unlink()
            log.info("removed scratch copy for %s", video_id)
    except OSError as exc:
        log.warning("could not remove scratch copy for %s (%s)", video_id, exc)
