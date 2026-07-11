"""SQLite persistence. One connection per call site (SQLite + WAL handles
concurrent readers; writes are short). JSON-heavy columns keep the schema
flexible while staying storage-light — we persist parameters, labels,
discrepancies, notes and feature traces, never derived video."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Any

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
  id TEXT PRIMARY KEY,
  filename TEXT NOT NULL,
  path TEXT NOT NULL,
  fps REAL, width INTEGER, height INTEGER, frame_count INTEGER,
  uploaded_at REAL NOT NULL,
  calibration_id TEXT,              -- clip-local override; NULL = use global
  learned_params_version INTEGER    -- version snapshot used at processing time
);

CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  video_id TEXT NOT NULL,
  kind TEXT NOT NULL,               -- 'pass1' | 'pass2'
  status TEXT NOT NULL,             -- 'queued' | 'running' | 'done' | 'error'
  progress REAL DEFAULT 0,
  message TEXT DEFAULT '',
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS rally_labels (
  id TEXT PRIMARY KEY,
  video_id TEXT NOT NULL,
  source TEXT NOT NULL,             -- 'human' | 'pass1' | 'pass2'
  start_frame INTEGER NOT NULL,
  end_frame INTEGER NOT NULL,
  meta TEXT DEFAULT '{}'            -- e.g. terminal event type, confidence
);

CREATE TABLE IF NOT EXISTS overlays (
  video_id TEXT NOT NULL,
  source TEXT NOT NULL,             -- 'pass1' | 'pass2'
  chunk_index INTEGER NOT NULL,     -- frames are chunked to keep rows small
  payload TEXT NOT NULL,            -- JSON: per-frame players/ball/events
  PRIMARY KEY (video_id, source, chunk_index)
);

CREATE TABLE IF NOT EXISTS calibrations (
  id TEXT PRIMARY KEY,
  scope TEXT NOT NULL,              -- 'global' | 'clip_local'
  video_id TEXT,                    -- set when clip_local
  corner_points TEXT NOT NULL,      -- JSON [{name, x, y}]
  homography TEXT NOT NULL,         -- JSON 3x3 (image -> court metres)
  focal_px REAL,                    -- estimated focal length, for ball depth
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS learned_params (
  version INTEGER PRIMARY KEY AUTOINCREMENT,
  params TEXT NOT NULL,             -- JSON dict of tunables
  origin TEXT NOT NULL,             -- 'default' | 'report:<id>' | 'manual'
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS user_notes (
  id TEXT PRIMARY KEY,
  video_id TEXT NOT NULL,
  text TEXT NOT NULL,
  source TEXT NOT NULL,             -- 'typed' | 'voice'
  scope TEXT NOT NULL,              -- 'global' | 'clip_local'
  video_time REAL,                  -- optional timestamp the note refers to
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS discrepancies (
  id TEXT PRIMARY KEY,
  video_id TEXT NOT NULL,
  report_id TEXT NOT NULL,
  human_rally TEXT,                 -- JSON {start_frame,end_frame} or NULL
  ai_rally TEXT,                    -- JSON {start_frame,end_frame} or NULL
  type TEXT NOT NULL,               -- 'boundary_drift' | 'missed' | 'extra'
  temporal_iou REAL,
  start_error INTEGER,              -- frames (ai - human); NULL if missed/extra
  end_error INTEGER,
  feature_trace TEXT                -- JSON per-frame trace around boundaries
);

CREATE TABLE IF NOT EXISTS reports (
  id TEXT PRIMARY KEY,
  video_id TEXT NOT NULL,
  ai_source TEXT NOT NULL,          -- which pass was compared
  summary TEXT NOT NULL,            -- JSON metrics summary
  llm_report TEXT,                  -- markdown from Anthropic (NULL if no key)
  llm_suggestions TEXT,             -- JSON structured param suggestions
  applied_params_version INTEGER,   -- LearnedParams version created from this
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS ball_corrections (
  id TEXT PRIMARY KEY,
  video_id TEXT NOT NULL,
  frame INTEGER NOT NULL,
  x REAL NOT NULL, y REAL NOT NULL,
  scope TEXT NOT NULL DEFAULT 'clip_local',
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS ball_tracks (
  video_id TEXT PRIMARY KEY,         -- one external ball track per clip (latest wins)
  payload TEXT NOT NULL,             -- JSON {frame: [x, y, diameter|null, confidence]} in full-res px
  source TEXT,                       -- which external tracker produced it (e.g. 'mti_v1')
  n_frames INTEGER,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS window_grades (
  id TEXT PRIMARY KEY,
  video_id TEXT NOT NULL,
  pass1_start_frame INTEGER NOT NULL, -- the graded Pass 1 window span; grades key
  pass1_end_frame INTEGER NOT NULL,   -- on the span, NOT pass1_window_id (re-minted
                                      -- every Pass 1 run), re-associated by IoU
  ai_verdict TEXT,                    -- 'rally'|'rejected' snapshot at grade time
  ai_rejection_reason TEXT,           -- 'no_serve'|'low_live_ratio'|'too_short'|NULL
  serve_gate_score REAL,              -- snapshot, for the recommendation sweep
  human_verdict TEXT NOT NULL,        -- 'keep' | 'drop' (human judgment, not "AI ok")
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS pending_changes (
  id TEXT PRIMARY KEY,
  video_id TEXT,                    -- clip that triggered it (context)
  param TEXT NOT NULL,
  from_value REAL,
  to_value REAL NOT NULL,
  risk TEXT NOT NULL,               -- 'medium' (low-risk auto-applies, never queued)
  rationale TEXT,
  evidence TEXT,                    -- JSON (evidence_n, harness before/after once decided)
  status TEXT NOT NULL,             -- 'pending'|'approved'|'rejected'|'rejected_by_harness'|'superseded'
  created_at REAL NOT NULL,
  decided_at REAL
);

CREATE TABLE IF NOT EXISTS param_eval (
  id TEXT PRIMARY KEY,
  version INTEGER,                  -- learned_params version this scores (NULL = rejected candidate)
  scope TEXT NOT NULL,              -- 'baseline'|'auto:label'|'approved'|'manual'
  mean_abs_start_err REAL,
  mean_abs_end_err REAL,
  match_rate REAL,
  n_clips INTEGER,
  n_matched INTEGER,
  n_extra INTEGER,
  applied INTEGER NOT NULL,         -- 1 = committed version, 0 = rejected candidate
  detail TEXT,                      -- JSON (before/after metrics, proposal, trigger)
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS contact_events (
  contact_id TEXT PRIMARY KEY,
  video_id TEXT NOT NULL,
  rally_id TEXT NOT NULL,
  frame_idx INTEGER NOT NULL,
  timestamp_s REAL NOT NULL,
  player_id INTEGER NOT NULL,
  team TEXT NOT NULL,               -- 'near' | 'far'
  court_x REAL, court_y REAL,
  shot_side TEXT,                   -- 'left' | 'right'
  hand_speed_peak REAL,
  confidence REAL,
  is_serve INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS coaching_reports (
  video_id TEXT PRIMARY KEY,        -- one report per video (latest wins, regenerate on request)
  category_ratios TEXT NOT NULL,    -- JSON {category: {flagged_count, active_rally_seconds, ratio}}
  worst_shots TEXT NOT NULL,        -- JSON [{contact_id, category, deviation_score, timestamp_s, rally_id, feature_vector}]
  best_shots TEXT NOT NULL,         -- JSON same shape as worst_shots
  narrative TEXT NOT NULL,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS model_registry (
  version INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,               -- 'rally_end' | 'serve_start' | 'tracknet'
  path TEXT NOT NULL,               -- data/models/<kind>/v<version>.pt
  status TEXT NOT NULL,             -- 'shadow' | 'live' | 'rejected' | 'archived'
  metrics TEXT,                     -- JSON held-out metrics (vs incumbent)
  parent_version INTEGER,
  trained_on TEXT,                  -- JSON {video_ids, n_windows, n_clips, label_hash}
  origin TEXT NOT NULL,             -- 'manual_cli' | 'auto_trigger'
  created_at REAL NOT NULL,
  promoted_at REAL
);
"""

# Columns added after the initial schema shipped. executescript won't ALTER an
# existing table, so add them idempotently on startup.
_ADDED_COLUMNS = [
    ("videos", "last_labels_hash", "TEXT"),
    ("videos", "model_versions", "TEXT"),
    # Ground-truth ball-event marks (viewer marking mode). NULL event_type =
    # legacy row = plain position mark. in_out is the human's call on a bounce;
    # in_out_suggested is what the homography said at save time; disagree=1
    # rows flag likely calibration error near lines.
    ("ball_corrections", "event_type", "TEXT"),
    ("ball_corrections", "court_x", "REAL"),
    ("ball_corrections", "court_y", "REAL"),
    ("ball_corrections", "in_out", "TEXT"),
    ("ball_corrections", "in_out_suggested", "TEXT"),
    ("ball_corrections", "disagree", "INTEGER DEFAULT 0"),
]


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init_db() -> None:
    with connect() as con:
        con.executescript(SCHEMA)
        for table, col, coltype in _ADDED_COLUMNS:
            have = {r["name"] for r in con.execute(f"PRAGMA table_info({table})")}
            if col not in have:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {coltype}")


def new_id() -> str:
    return uuid.uuid4().hex


def now() -> float:
    return time.time()


def rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]


def loads(s: str | None, default: Any = None) -> Any:
    if s is None:
        return default
    return json.loads(s)


def dumps(v: Any) -> str:
    return json.dumps(v, separators=(",", ":"))
