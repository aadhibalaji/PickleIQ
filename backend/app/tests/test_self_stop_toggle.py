"""Covers the self-stop toggle: default-on (unchanged behavior), persists
across get/set, and stop_this_instance() skips the EC2 call entirely (no
boto3/metadata-service reach) once disabled."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from .. import self_stop


@pytest.fixture(autouse=True)
def _isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(self_stop, "_STATE_PATH", tmp_path / "self_stop_state.json")
    yield


def test_defaults_enabled_when_never_set():
    assert self_stop.get_enabled() is True


def test_set_enabled_persists():
    self_stop.set_enabled(False)
    assert self_stop.get_enabled() is False
    self_stop.set_enabled(True)
    assert self_stop.get_enabled() is True


def test_stop_this_instance_skips_ec2_call_when_disabled():
    self_stop.set_enabled(False)
    with patch("requests.put") as mock_put:
        self_stop.stop_this_instance()
        mock_put.assert_not_called()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
