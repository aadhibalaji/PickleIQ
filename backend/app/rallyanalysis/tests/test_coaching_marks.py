"""coaching_marks: the expert ground-truth label store behind threshold
calibration (see backend/scripts/eval_coaching.py and the supervised sweep in
calibrate_thresholds.py).

The distinction these tests actually guard is the one that makes the table
worth having at all: spike-sourced marks are ONE verdict per spike (re-marking
replaces), while freeform marks append — an expert legitimately marks several
distinct bad spans in one stretch of play, and collapsing those would silently
destroy exactly the false-negative signal the table exists to capture.

Uses the same real-temp-sqlite fixture pattern as test_rally_source_default.py,
since this exercises the DB/API layer rather than pure functions."""
from __future__ import annotations

import pytest

from ... import db
from ...api import rally_analysis as api

VIDEO = "vid1"


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()
    with db.connect() as con:
        con.execute(
            "INSERT INTO videos (id, filename, path, fps, uploaded_at) VALUES (?,?,?,?,?)",
            (VIDEO, "v.mp4", "/tmp/v.mp4", 30.0, db.now()))
    yield


def _mark(**over):
    body = {"player_id": 1, "focus": "form", "start_frame": 10, "end_frame": 20,
            "verdict": "bad", "source": "freeform"}
    body.update(over)
    return body


def test_freeform_marks_accumulate():
    api.add_coaching_mark(VIDEO, _mark(start_frame=10, end_frame=20))
    api.add_coaching_mark(VIDEO, _mark(start_frame=40, end_frame=55))
    marks = api.list_coaching_marks(VIDEO)["marks"]
    assert len(marks) == 2
    assert [m["start_frame"] for m in marks] == [10, 40]  # ordered by start_frame


def test_remarking_a_spike_replaces_rather_than_duplicates():
    api.add_coaching_mark(VIDEO, _mark(source="spike", spike_id="spk_r1_1_knee_bend_deg_42",
                                        verdict="bad"))
    api.add_coaching_mark(VIDEO, _mark(source="spike", spike_id="spk_r1_1_knee_bend_deg_42",
                                        verdict="fine"))
    marks = api.list_coaching_marks(VIDEO)["marks"]
    assert len(marks) == 1
    assert marks[0]["verdict"] == "fine"


def test_distinct_spikes_do_not_collide():
    api.add_coaching_mark(VIDEO, _mark(source="spike", spike_id="spk_a"))
    api.add_coaching_mark(VIDEO, _mark(source="spike", spike_id="spk_b"))
    assert len(api.list_coaching_marks(VIDEO)["marks"]) == 2


def test_focus_filter_separates_form_and_positioning():
    api.add_coaching_mark(VIDEO, _mark(focus="form"))
    api.add_coaching_mark(VIDEO, _mark(focus="positioning"))
    assert len(api.list_coaching_marks(VIDEO, focus="form")["marks"]) == 1
    assert len(api.list_coaching_marks(VIDEO, focus="positioning")["marks"]) == 1
    assert len(api.list_coaching_marks(VIDEO)["marks"]) == 2


@pytest.mark.parametrize("bad", [
    {"verdict": "terrible"},          # not one of bad/fine/good
    {"focus": "tactics"},             # not a real focus
    {"source": "guess"},              # not a real source
    {"source": "spike"},              # spike source without a spike_id
    {"end_frame": 5},                 # end before start
    {"player_id": None},              # missing required int
])
def test_invalid_marks_are_rejected(bad):
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        api.add_coaching_mark(VIDEO, _mark(**bad))
    assert api.list_coaching_marks(VIDEO)["marks"] == []


def test_delete_removes_only_the_named_mark():
    keep = api.add_coaching_mark(VIDEO, _mark(start_frame=10, end_frame=20))
    drop = api.add_coaching_mark(VIDEO, _mark(start_frame=40, end_frame=55))
    api.delete_coaching_mark(drop["id"])
    marks = api.list_coaching_marks(VIDEO)["marks"]
    assert [m["id"] for m in marks] == [keep["id"]]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
