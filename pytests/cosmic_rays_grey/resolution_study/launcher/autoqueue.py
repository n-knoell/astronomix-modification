"""GPU queue of the resolution-study extension (H200 node): free GPUs only, at most MAX jobs.

Usage (venv python, detached):
    setsid nohup python launcher/autoqueue.py <queue file> [--max 2] > RAW/logs/autoqueue.log 2>&1 &

The queue file lists job tags, one per line (``#`` comments ignored); tag ``t`` runs
``RAW/jobs/t.sh`` with its log in ``RAW/logs/t.log``. The file is re-read every loop, so
tags can be appended while the queue runs; the queue exits when every listed tag has
finished and nothing runs.

Per job:
1. ``autocvd(interval=5, timeout=60)``, repeated, waits until a GPU has no compute process at
   all (any user), my running jobs' GPUs excluded; finished jobs are reaped between the waits;
2. the job starts pinned to that GPU's UUID (``CUDA_VISIBLE_DEVICES=GPU-...``, what
   ``common.require_pinned_gpu`` demands), ``XLA_PYTHON_CLIENT_MEM_FRACTION`` set by the job;
3. the queue waits until the job's python process holds that GPU (so the next selection
   cannot pick the same, still-empty GPU), then checks the PID -> UUID pin and that no
   other process shares the GPU. Wrong GPU or shared GPU: the job is killed and requeued.

State: ``RAW/logs/queue_events.txt`` (one line per event), ``RAW/logs/launches.txt``
(``verify.sh`` format), ``RAW/logs/done/<tag>`` with the exit code.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import autocvd

RAW = Path(os.environ.get("RES_STUDY_RAW", "/export/scratch/nknoell/resolution_study"))
LOGS = RAW / "logs"
DONE = LOGS / "done"


def event(msg):
    line = f"{time.strftime('%F %T')} {msg}"
    print(line, flush=True)
    with open(LOGS / "queue_events.txt", "a") as f:
        f.write(line + "\n")


def smi(*query):
    return subprocess.run(["nvidia-smi", *query, "--format=csv,noheader,nounits"],
                          capture_output=True, text=True).stdout


def gpu_uuids():
    """nvidia-smi index -> UUID (autocvd selects by nvidia-smi index)."""
    out = {}
    for line in smi("--query-gpu=index,uuid").splitlines():
        i, u = [x.strip() for x in line.split(",")]
        out[int(i)] = u
    return out


def compute_apps():
    """[(pid, gpu uuid)] of every compute process on the node."""
    apps = []
    for line in smi("--query-compute-apps=pid,gpu_uuid").splitlines():
        if line.strip():
            pid, u = [x.strip() for x in line.split(",")]
            apps.append((int(pid), u))
    return apps


def descendants(pid):
    out = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
    kids = [int(k) for k in out]
    return kids + [g for k in kids for g in descendants(k)]


def read_queue(path):
    tags = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if line and line not in tags:
            tags.append(line)
    return tags


def kill(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=60)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def launch(tag, uuid):
    log = open(LOGS / f"{tag}.log", "a")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": uuid}
    env.pop("CUDA_DEVICE_ORDER", None)
    proc = subprocess.Popen(["bash", str(RAW / "jobs" / f"{tag}.sh")], stdout=log, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)
    with open(LOGS / "launches.txt", "a") as f:
        f.write(f"{tag} pid={proc.pid} uuid={uuid} {time.strftime('%F %T')}\n")
    return proc


def confirm(proc, uuid, wait_s=600):
    """Wait until a python descendant of ``proc`` holds a GPU; 'ok', 'wrong', 'shared' or 'exited'."""
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if proc.poll() is not None:
            return "exited", []
        mine = set(descendants(proc.pid))
        apps = compute_apps()
        on = [u for p, u in apps if p in mine]
        if on:
            if any(u != uuid for u in on):
                return "wrong", on
            others = [p for p, u in apps if u == uuid and p not in mine]
            return ("shared" if others else "ok"), others
        time.sleep(5)
    return "timeout", []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("queue")
    ap.add_argument("--max", type=int, default=2)
    args = ap.parse_args()
    DONE.mkdir(parents=True, exist_ok=True)
    running = {}  # tag -> (proc, gpu index, uuid)
    requeue = []
    event(f"QUEUE START {args.queue} max={args.max} pid={os.getpid()}")
    while True:
        for tag, (proc, idx, uuid) in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                (DONE / tag).write_text(f"{rc}\n")
                event(f"END {tag} rc={rc} gpu={idx} {uuid}")
                del running[tag]
        started = set(running) | {p.name for p in DONE.iterdir()}
        pending = [t for t in requeue + read_queue(args.queue) if t not in started]
        pending = list(dict.fromkeys(pending))
        if not pending and not running:
            event("QUEUE DONE")
            return
        if not pending or len(running) >= args.max:
            time.sleep(5)
            continue
        tag = pending[0]
        if not (RAW / "jobs" / f"{tag}.sh").exists():
            event(f"SKIP {tag}: no jobs/{tag}.sh")
            (DONE / tag).write_text("missing\n")
            continue
        busy = [idx for _, idx, _ in running.values()]
        try:  # bounded wait, so finished jobs are reaped (and their GPUs no longer excluded)
            idx = autocvd.autocvd(num_gpus=1, interval=5, timeout=60, set_env=False, progress=False,
                                  exclude=busy or None)[0]
        except TimeoutError:
            continue
        uuid = gpu_uuids()[idx]
        # autocvd may have waited for hours: one of my jobs may have ended meanwhile, fine;
        # re-check that the GPU is still empty right before the launch
        if any(u == uuid for _, u in compute_apps()):
            continue
        proc = launch(tag, uuid)
        event(f"LAUNCH {tag} gpu={idx} {uuid} pid={proc.pid}")
        status, extra = confirm(proc, uuid)
        if status == "ok":
            event(f"VERIFIED {tag} on {uuid}")
            running[tag] = (proc, idx, uuid)
            if tag in requeue:
                requeue.remove(tag)
        elif status == "exited":
            rc = proc.returncode
            (DONE / tag).write_text(f"{rc}\n")
            event(f"END {tag} rc={rc} before holding a GPU (see logs/{tag}.log)")
        else:
            kill(proc)
            event(f"KILLED {tag}: {status} {extra}; requeued")
            if tag not in requeue:
                requeue.insert(0, tag)
            time.sleep(30)


if __name__ == "__main__":
    sys.exit(main())
