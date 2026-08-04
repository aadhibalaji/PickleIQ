"""Background job runner.

Processing is offline/asynchronous: uploads return immediately and Pass 1
starts in a worker thread, streaming results into the DB so the frontend can
poll while the user labels. A single worker keeps GPU/CPU contention sane on
a laptop; jobs queue behind it.

Because it's a single worker, a long job (a full 11-minute clip) blocks
shorter jobs queued behind it. So the runner supports:
  - cooperative cancellation (a long/unwanted job can be stopped to free the
    worker), and
  - superseding: re-queuing the same (video, kind) cancels the prior pending
    request instead of stacking a duplicate.
The API also reports each job's position in the queue so "queued" never looks
like "frozen".
"""
from __future__ import annotations

import logging
import queue
import threading
import traceback
from dataclasses import dataclass, field
from typing import Callable

from . import config, db, self_stop

log = logging.getLogger("jobs")


class JobCancelled(Exception):
    """Raised inside a job when it has been asked to stop."""


@dataclass
class Job:
    id: str
    video_id: str
    kind: str
    fn: Callable[["JobHandle"], None] = field(repr=False)
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    # Set by _loop when this job's timeout fires, so the (still-running,
    # orphaned) worker thread's own status write — whatever it eventually
    # does — doesn't clobber the 'error' status the timeout already recorded.
    timed_out: threading.Event = field(default_factory=threading.Event, repr=False)


class JobHandle:
    """Passed to job functions so they can report progress and check for
    cancellation."""

    def __init__(self, job: Job):
        self.job = job

    def cancelled(self) -> bool:
        return self.job.cancel.is_set()

    def raise_if_cancelled(self) -> None:
        if self.job.cancel.is_set():
            raise JobCancelled()

    def update(self, progress: float | None = None, message: str | None = None) -> None:
        # A cancelled job must stop promptly; the loops call this often.
        self.raise_if_cancelled()
        with db.connect() as con:
            if progress is not None:
                con.execute(
                    "UPDATE jobs SET progress=?, updated_at=? WHERE id=?",
                    (min(max(progress, 0.0), 1.0), db.now(), self.job.id),
                )
            if message is not None:
                con.execute(
                    "UPDATE jobs SET message=?, updated_at=? WHERE id=?",
                    (message, db.now(), self.job.id),
                )


class JobRunner:
    def __init__(self) -> None:
        self._q: queue.Queue[Job] = queue.Queue()
        self._jobs: dict[str, Job] = {}          # id -> Job (live)
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="job-runner")
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def submit(self, video_id: str, kind: str, fn: Callable[[JobHandle], None]) -> str:
        # Supersede any still-pending request for the same (video, kind).
        with self._lock:
            for j in self._jobs.values():
                if j.video_id == video_id and j.kind == kind:
                    j.cancel.set()
        with db.connect() as con:
            con.execute(
                "UPDATE jobs SET status='cancelled', message='superseded by a newer request',"
                " updated_at=? WHERE video_id=? AND kind=? AND status IN ('queued','running')",
                (db.now(), video_id, kind),
            )

        job_id = db.new_id()
        job = Job(id=job_id, video_id=video_id, kind=kind, fn=fn)
        with self._lock:
            self._jobs[job_id] = job
        with db.connect() as con:
            con.execute(
                "INSERT INTO jobs (id, video_id, kind, status, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?)",
                (job_id, video_id, kind, "queued", db.now(), db.now()),
            )
        self._q.put(job)
        return job_id

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return False
        job.cancel.set()
        with db.connect() as con:
            # If it's still queued it may never start, so mark it now; a
            # running job will finalize itself when it notices the flag.
            con.execute(
                "UPDATE jobs SET status='cancelled', message='cancelled',"
                " updated_at=? WHERE id=? AND status='queued'",
                (db.now(), job_id),
            )
        return True

    def cancel_video(self, video_id: str) -> int:
        with self._lock:
            ids = [j.id for j in self._jobs.values() if j.video_id == video_id]
        for jid in ids:
            self.cancel(jid)
        return len(ids)

    def queue_positions(self) -> dict[str, int]:
        """job_id -> number of jobs ahead of it in the queue (0 = next up)."""
        with self._lock:
            pending = [j.id for j in list(self._q.queue) if not j.cancel.is_set()]
        return {jid: i for i, jid in enumerate(pending)}

    def _set_status(self, job_id: str, status: str, message: str = "") -> None:
        with db.connect() as con:
            con.execute(
                "UPDATE jobs SET status=?, message=?, updated_at=? WHERE id=?",
                (status, message, db.now(), job_id),
            )

    def _run_job_body(self, job: Job) -> None:
        """Runs job.fn and writes its terminal DB status. Split out of _loop
        so it can run on its own thread — _loop then bounds it with
        Thread.join(timeout) instead of blocking the single worker forever on
        a hung job.fn (see module docstring: a job once hung ~22.8h with zero
        output). Skips writing 'done'/'cancelled'/'error' if the timeout
        already fired for this job (job.timed_out) — that status is final."""
        try:
            job.fn(JobHandle(job))
            if not job.timed_out.is_set():
                self._set_status(job.id, "done")
        except JobCancelled:
            if not job.timed_out.is_set():
                self._set_status(job.id, "cancelled", "cancelled")
                log.info("job %s (%s) cancelled", job.id, job.kind)
        except Exception as exc:  # noqa: BLE001 — job errors must not kill the worker
            if job.timed_out.is_set():
                log.info("job %s (%s) raised after timing out (%s) — ignored, "
                          "status already finalized", job.id, job.kind, exc)
            else:
                log.error("job %s (%s) failed:\n%s", job.id, job.kind, traceback.format_exc())
                self._set_status(job.id, "error", f"{type(exc).__name__}: {exc}")

    def _timeout_for(self, kind: str) -> float:
        if kind == "pass1":
            return config.JOB_TIMEOUT_PASS1_S
        if kind == "pass2":
            return config.JOB_TIMEOUT_PASS2_S
        if kind.startswith("pass2_range_"):
            return config.JOB_TIMEOUT_PASS2_RANGE_S
        if kind == "contacts":
            return config.JOB_TIMEOUT_CONTACTS_S
        if kind == "learn":
            return config.JOB_TIMEOUT_LEARN_S
        return config.JOB_TIMEOUT_DEFAULT_S

    def _on_timeout(self, job: Job) -> None:
        message = "job exceeded timeout, likely hung"
        self._set_status(job.id, "error", message)
        log.error("job %s (%s) timed out — marked failed, queue continues", job.id, job.kind)
        if job.kind.startswith("pass2_range_"):
            # pass2_range jobs are keyed 1:1 with a coaching_selections row
            # (see api/rally_analysis.py's process_selection) — that row must
            # not be left stuck on 'processing' just because the job that was
            # supposed to finish it hung. Lazy import: pipeline imports
            # jobs.runner at module load, so a top-level import here would
            # be circular.
            from . import pipeline
            selection_id = job.kind[len("pass2_range_"):]
            pipeline._set_selection_status(selection_id, "failed", message)

    def _has_pending_work(self) -> bool:
        """DB-backed, independent of self._q's own in-memory state. submit()
        writes the jobs row (status='queued') BEFORE putting the Job onto
        self._q — correct today, but it means self._q.empty() alone is not
        a sufficient "nothing left to run" guarantee: it's trusting that no
        code path ever inserts a row without also enqueuing it, forever.
        Querying the jobs table directly checks the durable, authoritative
        record instead of that invariant holding by convention — see
        _loop's self-stop gate below for why this matters."""
        with db.connect() as con:
            row = con.execute(
                "SELECT 1 FROM jobs WHERE status IN ('queued','running') LIMIT 1").fetchone()
        return row is not None

    def _loop(self) -> None:
        while True:
            job = self._q.get()
            try:
                if job.cancel.is_set():
                    self._set_status(job.id, "cancelled", "cancelled before start")
                    continue
                self._set_status(job.id, "running")
                worker = threading.Thread(target=self._run_job_body, args=(job,),
                                           daemon=True, name=f"job-{job.kind}")
                worker.start()
                worker.join(self._timeout_for(job.kind))
                if worker.is_alive():
                    job.timed_out.set()
                    job.cancel.set()  # best-effort nudge — see JobHandle.update()
                    self._on_timeout(job)
                    # ponytail: the orphaned worker thread keeps running
                    # detached in the background (no safe hard-kill for
                    # threads in Python) and may still write DB rows/overlay
                    # data after this point. Acceptable for a "don't hang
                    # forever silently" safeguard; upgrade to a
                    # process-per-job model (real kill) if stray writes from
                    # a timed-out job ever bite in practice.
            finally:
                with self._lock:
                    self._jobs.pop(job.id, None)

            # Chained Pass1/Pass2/contacts steps for one video all run
            # INSIDE job.fn() above (each stage's own function body calls
            # runner.submit() for the next stage before returning — see
            # pipeline.py's do_pass1/_build_do_pass2), so in the designed
            # case this only ever fires between distinct queue entries,
            # never mid-chain. self._q.empty() is checked first as a cheap
            # short-circuit (skips the DB query on every ordinary busy
            # tick); _has_pending_work() is the actual authority — see its
            # docstring for why the two can't be assumed to always agree.
            if self._q.empty() and not self._has_pending_work():
                self_stop.stop_this_instance()


runner = JobRunner()
