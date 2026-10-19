"""Tiny in-memory background job runner (thread pool).  Results are lost on restart, by design.

Usage:   job_id = jobs.submit(fn, "screen")        # fn(job) -> JSON-serialisable result
         jobs.status(job_id)  -> {"state", "progress", "message", "result"?, "error"?}
Inside fn, call job.update(fraction, "message") to report progress."""
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

TTL_SECONDS = 3600


class Job:
    def __init__(self, kind):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.state = "queued"           # queued | running | done | error
        self.progress = 0.0
        self.message = ""
        self.result = None
        self.error = None
        self.created = time.time()

    def update(self, progress=None, message=None):
        if progress is not None:
            self.progress = max(0.0, min(1.0, float(progress)))
        if message is not None:
            self.message = message


class JobManager:
    def __init__(self, workers=2):
        self.pool = ThreadPoolExecutor(max_workers=max(1, int(workers)))
        self.jobs = {}
        self.lock = threading.Lock()

    def submit(self, fn, kind="job"):
        self._cleanup()
        job = Job(kind)
        with self.lock:
            self.jobs[job.id] = job

        def runner():
            job.state = "running"
            try:
                job.result = fn(job)
                job.progress, job.state = 1.0, "done"
            except Exception as e:                         # shown to the user; never crashes the server
                job.error = str(e) or e.__class__.__name__
                job.state = "error"
                traceback.print_exc()

        self.pool.submit(runner)
        return job.id

    def get(self, job_id):
        with self.lock:
            return self.jobs.get(job_id)

    def status(self, job_id):
        job = self.get(job_id)
        if job is None:
            return None
        d = {"id": job.id, "kind": job.kind, "state": job.state, "progress": round(job.progress, 3),
             "message": job.message, "error": job.error}
        if job.state == "done":
            d["result"] = job.result
        return d

    def _cleanup(self):
        cutoff = time.time() - TTL_SECONDS
        with self.lock:
            for jid in [j for j, job in self.jobs.items() if job.created < cutoff]:
                del self.jobs[jid]
