"""_sqs_loop must route each S3-event key to the same handler the bucket-scan
backstop uses, and must only ack (delete) a message once every key in it was
handled without raising — a partial failure should leave the message for
SQS's own redelivery, not get silently dropped.

_sqs_loop is a `while True`, so each test breaks out of it with a
BaseException subclass raised from receive_job_events — deliberately NOT an
Exception, since the loop's own `except Exception` (which every real AWS
hiccup should hit) must not also swallow the test's exit signal."""
from unittest.mock import patch

from .. import cloud_ingest


class _StopLoop(BaseException):
    pass


def _run_until_stopped(**patches):
    with patch.object(cloud_ingest, "SQS_JOB_QUEUE_URL", "fake-queue-url"), \
         patch.object(cloud_ingest, "_known_video_ids", return_value=set()), \
         patch.multiple(cloud_ingest, **patches):
        try:
            cloud_ingest._sqs_loop()
        except _StopLoop:
            pass


def test_routes_video_key_to_video_handler():
    calls = []
    key = "default/" + "a" * 32 + ".mp4"
    receive = _mock_receive_once([{"receipt_handle": "r1", "keys": [key]}])
    _run_until_stopped(
        _handle_video_key=lambda k, known: calls.append(("video", k)),
        storage_s3=_FakeStorageS3(receive, lambda rh: calls.append(("ack", rh))),
    )
    assert ("video", key) in calls
    assert ("ack", "r1") in calls


def test_partial_failure_does_not_ack():
    calls = []

    def _boom(key, known):
        raise RuntimeError("boom")

    key = "default/" + "b" * 32 + ".mp4"
    receive = _mock_receive_once([{"receipt_handle": "r1", "keys": [key]}])
    _run_until_stopped(
        _handle_video_key=_boom,
        storage_s3=_FakeStorageS3(receive, lambda rh: calls.append(("ack", rh))),
    )
    assert calls == []  # never ack'd — message stays for SQS redelivery


def _mock_receive_once(first_batch):
    calls = {"n": 0}

    def receive(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return first_batch
        raise _StopLoop
    return receive


class _FakeStorageS3:
    """Swaps in for cloud_ingest.storage_s3 for the duration of one test —
    only the two methods _sqs_loop actually calls need to exist."""
    def __init__(self, receive_job_events, delete_job_event):
        self.receive_job_events = receive_job_events
        self.delete_job_event = delete_job_event


if __name__ == "__main__":
    test_routes_video_key_to_video_handler()
    test_partial_failure_does_not_ack()
    print("ok")
