"""Write the extension-D job scripts (RAW/jobs/D_<tag>.sh) from D0_sizing.json and print the tags in
queue order (PLAN.md Section 9.5). Usage: python mk_d_jobs.py [group ...]; groups: smoke, d0, wind, sedov,
cwb, d5 (default all, in that order)."""

import json
import sys
from pathlib import Path

RAW = Path("/export/scratch/nknoell/resolution_study")
HERE = Path(__file__).resolve().parents[1]
plan = json.loads((HERE / "D0_sizing.json").read_text())

HEAD = """#!/bin/bash
source ~/venv_grav_source/bin/activate
export PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
cd {here}
echo "host $(hostname) CVD=$CUDA_VISIBLE_DEVICES"
rc=0
"""


def job(tag, cmd):
    body = HEAD.format(here=HERE) + (f'echo "=== {cmd} $(date +%T)"; {cmd} || {{ rc=$?; echo "FAILED rc=$rc"; }}\n'
                                     f'echo JOBDONE {tag} $(date +%T)\nexit $rc\n')
    path = RAW / "jobs" / f"{tag}.sh"
    path.write_text(body)
    path.chmod(0o755)
    return tag


def lam_tag(lam):
    return f"lam{lam:g}"


def wind(n, lam=None, vred=None, extra="", suffix=""):
    if lam is None:
        return job(f"D_w_adv{suffix}_N{n}", f"python wind_study.py --n {n} --name adv{suffix} --vred 0 {extra}".strip())
    p = plan[f"wind_lam{lam:g}"]
    v = vred or p["vred"]
    return job(f"D_w_{lam_tag(lam)}{suffix}_N{n}",
               f"python wind_study.py --n {n} --name {lam_tag(lam)}{suffix} --kappa {p['kappa']} --vred {v} {extra}".strip())


def sedov(setup, n, lam, kr13=False, vred=None, suffix=""):
    p = plan[f"{setup}_lam{lam:g}"]
    v = vred or p["vred"]
    model = "kr13" if kr13 else "z0.1"
    stop = f" --stop-radius {p['stop_radius']}" if p["stop_radius"] else ""
    return job(f"D_{setup}_{model}_{lam_tag(lam)}{suffix}_N{n}",
               f"python sedov_study.py --setup {setup} --n {n} " + ("--kr13" if kr13 else "--zeta 0.1")
               + f" --kappa {p['kappa']} --vred {v} --name {model}_{lam_tag(lam)}{suffix}{stop}")


def cwb(n, lam, vred=None, suffix=""):
    p = plan[f"cwb_lam{lam:g}"]
    v = vred or p["vred"]
    return job(f"D_cwb_{lam_tag(lam)}{suffix}_N{n}",
               f"python cwb_study.py --n {n} --zeta 0.1 --kappa {p['kappa']} --vred {v} --name z0.1_{lam_tag(lam)}{suffix}")


LAMS = (0.03, 0.1, 0.3)
SEDOV_NS = (48, 64, 96, 128, 192)
WIND_NS = (64, 96, 128, 160, 192)
CWB_NS = (64, 96, 128, 192)

groups = {
    # one small run per script: checks the new code paths before the ladders
    "smoke": lambda: [wind(64, 0.1), sedov("ss", 48, 0.1), sedov("cp", 48, 0.1, kr13=True), cwb(64, 0.1)],
    # D0 checks: v_red x2 at N 96, S-PE at v_red 0, self re-acceleration probe
    "d0": lambda: [
        wind(96, 0.1, vred=2 * plan["wind_lam0.1"]["vred"], suffix="_vx2"),
        sedov("ss", 96, 0.1, vred=2 * plan["ss_lam0.1"]["vred"], suffix="_vx2"),
        sedov("cp", 96, 0.1, vred=2 * plan["cp_lam0.1"]["vred"], suffix="_vx2"),
        cwb(96, 0.1, vred=2 * plan["cwb_lam0.1"]["vred"], suffix="_vx2"),
        job("D_pe_vred0_N128", "python sedov_study.py --setup pe --n 128 --zeta 0.1 --vred 0 --name z0.1_vred0"),
        wind(128, 0.1, extra="--off-after 0.5", suffix="_off05"),
        wind(128, None, extra="--off-after 0.5", suffix="_off05"),
    ],
    # D1: largest first so the long runs start early
    "wind": lambda: [t for n in sorted(WIND_NS, reverse=True)
                     for t in [wind(n)] + [wind(n, lam) for lam in LAMS] if not (n == 64 and t == "D_w_lam0.1_N64")],
    "sedov": lambda: [sedov(s, n, lam, kr13=k) for n in sorted(SEDOV_NS, reverse=True) for s, k in
                      (("cp", False), ("cp", True), ("ss", False)) for lam in LAMS
                      if not (n == 48 and lam == 0.1 and (s, k) in (("ss", False), ("cp", True)))],
    "cwb": lambda: [cwb(n, lam) for n in sorted(CWB_NS, reverse=True) for lam in LAMS
                    if not (n == 64 and lam == 0.1)],
    # D5 (H200, RESULTS.md D4 "Input for D5"); CWB excluded (user 2026-11-06), S-SS waits for DB13.
    # Not launched from here: H200 rules of EXTENSION_KICKSTART.md (NATIVE_JAX, max 2 GPUs).
    "d5": lambda: ([t for n in (320, 256) for t in [wind(n), wind(n, 0.03), wind(n, 0.3)]]
                   + [sedov("cp", n, lam, kr13=k) for n in (384, 256) for k in (False, True) for lam in (0.03, 0.1)]),
}
for g in (sys.argv[1:] or list(groups)):
    print(" ".join(groups[g]()))
