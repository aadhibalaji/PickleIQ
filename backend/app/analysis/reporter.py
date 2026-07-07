"""Discrepancy feedback report via the Anthropic API.

The LLM is used ONLY here — never in the per-frame vision loop. It receives
the structured discrepancy data (predicted vs human pairs, per-frame feature
traces around each disputed boundary, and the user's free-form notes),
explains WHY the discrepancies likely occurred and HOW to fix them, and may
propose LearnedParams adjustments. Proposed adjustments are clamped through
learning.params.BOUNDS before ever touching the global baseline.
"""
from __future__ import annotations

import logging

from .. import db
from ..config import ANTHROPIC_API_KEY, ANTHROPIC_REPORT_MODEL
from ..learning.params import BOUNDS, DEFAULTS

log = logging.getLogger("analysis.reporter")

REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "report_markdown": {
            "type": "string",
            "description": "The full feedback report in markdown: per-discrepancy "
                           "likely cause, routed by type (boundary_drift vs "
                           "missed/extra), and concrete fixes.",
        },
        "param_suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "param": {"type": "string"},
                    "value": {"type": "number"},
                    "rationale": {"type": "string"},
                    "scope": {"type": "string", "enum": ["global", "clip_local"]},
                },
                "required": ["param", "value", "rationale", "scope"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["report_markdown", "param_suggestions"],
    "additionalProperties": False,
}

SYSTEM = """You are the analysis engine of a pickleball rally-detection system.

The system detects rallies (serve to the frame the ball becomes unplayable) in
fixed-camera videos using a two-pass pipeline:
- Pass 1: coarse gate over 100% of footage (motion energy, coarse ball blobs,
  sparse serve-formation checks) -> candidate windows, padded ~1-2 s.
- Pass 2: inside candidate windows only — full-rate ball tracking, 3D height
  reconstruction from apparent ball size + homography (axial camera view, so
  bounces are height minima, not 2D "V"s), bounce-vs-hit classification, and
  terminal-event end logic (bounce out / double bounce / net fail), confirmed
  but never timed by player behaviour.

You receive human ground-truth labels vs AI detections with temporal IoU and
start/end errors in frames, per-frame feature traces around disputed
boundaries, the user's free-form notes, and the current tunable parameters.

Diagnose WHY each discrepancy likely occurred and HOW to fix it. Route by
type: boundary_drift means the rally was found but boundaries are off
(usually threshold/offset issues); missed/extra means the Pass 1 gate or
minimum-duration filter is wrong. Systematic errors (consistent sign across
rallies) deserve parameter changes; one-off errors usually do not. Only
suggest parameters from the provided list, and mark a suggestion clip_local
if it stems from an oddity of this specific clip rather than the method."""


def generate_llm_report(payload: dict) -> tuple[str | None, list[dict]]:
    """Returns (report_markdown, clamped param_suggestions). Degrades to
    (None, []) with a logged reason when the API key is missing or the call
    fails — the deterministic metrics/updates still work without it."""
    if not ANTHROPIC_API_KEY:
        log.warning("ANTHROPIC_API_KEY not set — skipping LLM report")
        return None, []
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    param_doc = "\n".join(
        f"- {k} (default {v}, bounds {BOUNDS.get(k)})" for k, v in DEFAULTS.items()
    )
    try:
        response = client.messages.create(
            # Cost controls (the report was the pipeline's only real token sink —
            # Pass 1/Pass 2 are pure local CV): a cheaper model, a *capped*
            # thinking budget instead of adaptive (which could spend a large,
            # variable amount), and a max_tokens ceiling sized to the actual JSON
            # report rather than 16k. Together ~5-10x cheaper per report.
            model=ANTHROPIC_REPORT_MODEL,
            max_tokens=6000,
            thinking={"type": "enabled", "budget_tokens": 2000},
            system=SYSTEM,
            output_config={"format": {"type": "json_schema", "schema": REPORT_SCHEMA}},
            messages=[{
                "role": "user",
                "content": "Tunable parameters:\n" + param_doc
                           + "\n\nDiscrepancy data (JSON):\n" + db.dumps(payload),
            }],
        )
    except anthropic.RateLimitError:
        log.error("Anthropic rate limit — try Generate report again shortly")
        return None, []
    except anthropic.AuthenticationError:
        log.error("Anthropic API key invalid — check .env")
        return None, []
    except anthropic.APIStatusError as exc:
        log.error("Anthropic API error %s: %s", exc.status_code, exc.message)
        return None, []
    except anthropic.APIConnectionError:
        log.error("network error reaching the Anthropic API")
        return None, []

    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        data = db.loads(text, {})
    except Exception:  # noqa: BLE001
        log.error("LLM returned unparseable JSON")
        return None, []
    report = data.get("report_markdown")
    suggestions = []
    for s in data.get("param_suggestions", []):
        if s.get("param") in DEFAULTS:
            suggestions.append(s)  # clamping happens in learning.params.apply_updates
    return report, suggestions
