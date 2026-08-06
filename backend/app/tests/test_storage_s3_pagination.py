"""_list_all_keys must follow S3's pagination — a plain list_objects_v2 call
silently caps at 1000 keys/response, which was the actual root cause of the
cloud worker never noticing new video uploads once a bucket grew past 1000
video objects (list_video_keys, list_pass2_request_video_ids,
list_kill_request_video_ids all went through it)."""
from unittest.mock import patch

from .. import storage_s3


class _FakePaginator:
    def __init__(self, pages):
        self._pages = pages

    def paginate(self, **_kwargs):
        return self._pages


class _FakeClient:
    def __init__(self, pages):
        self._pages = pages

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _FakePaginator(self._pages)


def test_list_all_keys_spans_multiple_pages():
    page1 = {"Contents": [{"Key": f"default/{i}.mp4"} for i in range(1000)]}
    page2 = {"Contents": [{"Key": "default/1000.mp4"}]}
    with patch.object(storage_s3, "_client", return_value=_FakeClient([page1, page2])), \
         patch.object(storage_s3, "S3_VIDEO_BUCKET", "fake-bucket"):
        keys = storage_s3._list_all_keys("default/")
    assert len(keys) == 1001
    assert "default/1000.mp4" in keys


if __name__ == "__main__":
    test_list_all_keys_spans_multiple_pages()
    print("ok")
