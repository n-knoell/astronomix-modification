"""GPU queue that keeps its GPUs between jobs (H200 node): free GPUs only, at most MAX GPUs.

Usage (venv python, detached):
    setsid nohup python launcher/gpu_lanes.py <queue file> [<queue file> ...] [--max 2] \
        > RAW/logs/gpu_lanes.log 2>&1 &

Same job/queue conventions as ``autoqueue.py`` (tag ``t`` runs ``RAW/jobs/t.sh``, log
``RAW/logs/t.log``, done marker ``RAW/logs/done/t``, events in ``RAW/logs/queue_events.txt``);
the queue files are read in order and re-read every loop.

Difference: a GPU is taken once and then kept (a "lane"). When a lane is acquired, a tiny
holder process (one CUDA context + 1 MiB, ~0.5 GB) is started on it and stays there until the
lane is released, so the GPU never looks free between two jobs; the next job starts on the
same GPU as soon as the previous one ends.

Per lane:
1. acquire: ``autocvd(interval=5, timeout=30)`` finds a GPU without any compute process (any
   user), my lanes excluded; the holder starts pinned to its UUID; the lane is kept only if the
   holder is then alone on that GPU;
2. jobs run one after another on the lane (UUID pin), each confirmed as in ``autoqueue.py``
   (its python holds the lane's GPU, nothing else there but my holder; else killed + requeued);
3. before each launch the lane is checked: if another user's process sits on it, the lane is
   dropped (holder stopped) and a new GPU is acquired;
4. release: when no job is left for it. The holder also exits by itself if this dispatcher dies.

Lanes are acquired only while there are more pending jobs than idle lanes.
"""

import argparse
import ctypes
import os
import subprocess
import sys
import time

import autocvd

from autoqueue import DONE, RAW, compute_apps, descendants, event, gpu_uuids, kill, launch, read_queue


def hold():
    """Holder child: one CUDA context + 1 MiB on the pinned GPU until the parent goes away."""
    parent = os.getppid()
    cu = ctypes.CDLL("libcuda.so.1")
    dev, ctx, ptr = ctypes.c_int(), ctypes.c_void_p(), ctypes.c_uint64()
    for name, call in (("cuInit", lambda: cu.cuInit(0)),
                       ("cuDeviceGet", lambda: cu.cuDeviceGet(ctypes.byref(dev), 0)),
                       ("cuCtxCreate", lambda: cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)),
                       ("cuMemAlloc", lambda: cu.cuMemAlloc_v2(ctypes.byref(ptr), ctypes.c_size_t(1 << 20)))):
        err = call()
        if err:
            print(f"holder: {name} failed ({err})", flush=True)
            return 1
    print(f"holder {os.getpid()} on {os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
    while os.getppid() == parent:
        time.sleep(5)
    return 0


def confirm(proc, uuid, mine_extra=(), wait_s=600):
    """``autoqueue.confirm`` with my holder counted as mine: 'ok', 'wrong', 'shared', 'exited', 'timeout'."""
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
            others = [p for p, u in apps if u == uuid and p not in mine and p not in mine_extra]
            return ("shared" if others else "ok"), others
        time.sleep(5)
    return "timeout", []


def start_holder(uuid):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": uuid}
    env.pop("CUDA_DEVICE_ORDER", None)
    log = open(RAW / "logs" / "gpu_lanes_holders.log", "a")
    return subprocess.Popen([sys.executable, os.path.abspath(__file__), "--hold"], stdout=log,
                            stderr=subprocess.STDOUT, env=env, start_new_session=True)


def stop(proc):
    if proc.poll() is None:
        kill(proc)


def acquire(busy):
    """A new lane {idx, uuid, holder} on a GPU with no compute process, or None."""
    try:
        idx = autocvd.autocvd(num_gpus=1, interval=5, timeout=30, set_env=False, progress=False,
                              exclude=busy or None)[0]
    except TimeoutError:
        return None
    uuid = gpu_uuids()[idx]
    if any(u == uuid for _, u in compute_apps()):
        return None
    holder = start_holder(uuid)
    t0 = time.time()
    while time.time() - t0 < 120:
        if holder.poll() is not None:
            event(f"LANE FAIL gpu={idx} {uuid}: holder exited rc={holder.returncode} (logs/gpu_lanes_holders.log)")
            return None
        apps = compute_apps()
        if any(p == holder.pid for p, _ in apps):
            on = [u for p, u in apps if p == holder.pid]
            others = [p for p, u in apps if u == uuid and p != holder.pid]
            if on != [uuid] or others:
                stop(holder)
                event(f"LANE REJECT gpu={idx} {uuid}: holder on {on}, others {others}")
                time.sleep(30)
                return None
            event(f"LANE ACQUIRED gpu={idx} {uuid} holder={holder.pid}")
            return {"idx": idx, "uuid": uuid, "holder": holder, "job": None}
        time.sleep(2)
    stop(holder)
    event(f"LANE FAIL gpu={idx} {uuid}: holder not on the GPU after 120 s")
    return None


def release(lane, why):
    stop(lane["holder"])
    event(f"LANE RELEASED gpu={lane['idx']} {lane['uuid']} ({why})")


def main():
    if "--hold" in sys.argv:
        return hold()
    ap = argparse.ArgumentParser()
    ap.add_argument("queues", nargs="+")
    ap.add_argument("--max", type=int, default=2)
    args = ap.parse_args()
    DONE.mkdir(parents=True, exist_ok=True)
    lanes, requeue = [], []
    event(f"LANES START {' '.join(args.queues)} max={args.max} pid={os.getpid()}")
    try:
        while True:
            # reap finished jobs; a lane whose holder died is dropped once idle
            for lane in list(lanes):
                if lane["job"] is not None:
                    tag, proc = lane["job"]
                    rc = proc.poll()
                    if rc is not None:
                        (DONE / tag).write_text(f"{rc}\n")
                        event(f"END {tag} rc={rc} gpu={lane['idx']} {lane['uuid']}")
                        lane["job"] = None
                if lane["job"] is None and lane["holder"].poll() is not None:
                    lanes.remove(lane)
                    event(f"LANE LOST gpu={lane['idx']} {lane['uuid']}: holder exited")
            running = {lane["job"][0] for lane in lanes if lane["job"] is not None}
            started = running | {p.name for p in DONE.iterdir()}
            listed = [t for q in args.queues for t in read_queue(q)]
            pending = list(dict.fromkeys(t for t in requeue + listed if t not in started))
            idle = [lane for lane in lanes if lane["job"] is None]
            if not pending:
                for lane in idle:
                    lanes.remove(lane)
                    release(lane, "no job left")
                if not lanes:
                    event("LANES DONE")
                    return 0
                time.sleep(5)
                continue
            # fill idle lanes
            for lane in idle:
                if not pending:
                    break
                foreign = [p for p, u in compute_apps() if u == lane["uuid"] and p != lane["holder"].pid]
                if foreign:
                    lanes.remove(lane)
                    release(lane, f"other process(es) {foreign} on it")
                    continue
                tag = pending.pop(0)
                if not (RAW / "jobs" / f"{tag}.sh").exists():
                    event(f"SKIP {tag}: no jobs/{tag}.sh")
                    (DONE / tag).write_text("missing\n")
                    continue
                proc = launch(tag, lane["uuid"])
                event(f"LAUNCH {tag} gpu={lane['idx']} {lane['uuid']} pid={proc.pid} (lane)")
                status, extra = confirm(proc, lane["uuid"], mine_extra={lane["holder"].pid})
                if status == "ok":
                    event(f"VERIFIED {tag} on {lane['uuid']}")
                    lane["job"] = (tag, proc)
                    if tag in requeue:
                        requeue.remove(tag)
                elif status == "exited":
                    (DONE / tag).write_text(f"{proc.returncode}\n")
                    event(f"END {tag} rc={proc.returncode} before holding a GPU (see logs/{tag}.log)")
                else:
                    kill(proc)
                    event(f"KILLED {tag}: {status} {extra}; requeued")
                    if tag not in requeue:
                        requeue.insert(0, tag)
                    pending.insert(0, tag)
            # more work than idle lanes: try to take another GPU
            idle = [lane for lane in lanes if lane["job"] is None]
            if len(lanes) < args.max and len(pending) > len(idle):
                lane = acquire([lane["idx"] for lane in lanes])
                if lane is not None:
                    lanes.append(lane)
                continue
            time.sleep(5)
    finally:
        for lane in lanes:
            if lane["job"] is None:
                release(lane, "dispatcher exit")


if __name__ == "__main__":
    sys.exit(main())
