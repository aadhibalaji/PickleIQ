"""Repo-root resolution, shared by config.py / hw_profile.py / refine/*.

Computed as two levels above this file (backend/app/paths.py -> backend/app
-> backend -> repo root) — every other file that needs the root imports it
from here instead of recomputing its own parents[N] walk, so a file moving
to a different depth (e.g. refine/config.py) can't silently end up with a
different answer than config.py.

REPO_ROOT env var overrides the computed value outright. The Dockerfile sets
it explicitly (ENV REPO_ROOT=/app) so container correctness doesn't depend
on this relative walk lining up with whatever COPY layout the image uses.
"""
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    override = os.getenv("REPO_ROOT")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2]
