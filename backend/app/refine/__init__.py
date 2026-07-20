"""Pass-2 rally refinement.

Consumes (a) Pass 1 candidate windows (``rally_labels source='pass1'``) and
(b) the per-frame tracking data inside those windows (the ``pass2`` overlay:
player court positions, ball position, bounce/hit events) and emits, per
candidate window, either a rejection or refined start/end boundaries.

Never re-runs any vision model and never scans outside the Pass 1 windows
(plus a small configurable margin for serve-gate lookback). Pure numpy over
already-extracted tracking data — a full clip refines in well under a second
on CPU.

Output is JSON-only for now (``data/refine/<video_id>.json``); it writes no
``rally_labels`` rows, so Pass 1 stays the app's boundary source until the
refiner is validated.
"""
from .refine import refine_video, run_refine_json  # noqa: F401
