"""H200 extension (EXTENSION_KICKSTART.md): write every job script (RAW/jobs/H_<tag>.sh) and the queue
files RAW/jobs/queue_h200_<phase>.txt, and print a cost table (2080 Ti hours, scaled ~N^4 from the
finest existing run of the same kind; the H200 speed-up is measured in phase 0).

Phases (feed them to ``launcher/autoqueue.py`` one after the other, --max 2):
  0  overlap checks + memory probes (decide 512^3 and confirm the H200 runs match the 2080 Ti runs)
  1  advective / hydro ladders (the original study's setups at higher N; also the lambda = 0 anchors of 2)
  2  CR diffusion (extension D, D5): W, S-CP, S-SS (S-SS also as the DB13 probe at higher N)
  3  optional: Sedov 512^3 (only if phase 0 says it fits), CWB KR13 (DB12)
Within a phase the jobs are ordered by N ascending, so a partial queue still gives complete lower rungs.

Usage: python launcher/mk_h200_jobs.py   (writes everything, prints the table)
"""

import json
from pathlib import Path

import numpy as np

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


def wall_h(path):
    p = RAW / path
    return float(np.load(p)["wall"].sum()) / 3600 if p.exists() else np.nan


jobs = {0: [], 1: [], 2: [], 3: []}


def job(phase, tag, cmds, ref=None, n=None, n_ref=None):
    body = HEAD.format(here=HERE)
    for c in cmds:
        body += f'echo "=== {c} $(date +%T)"; {c} || {{ rc=$?; echo "FAILED rc=$rc: {c}"; }}\n'
    body += f"echo JOBDONE {tag} $(date +%T)\nexit $rc\n"
    path = RAW / "jobs" / f"{tag}.sh"
    path.write_text(body)
    path.chmod(0o755)
    est = wall_h(ref) * (n / n_ref) ** 4 if ref else np.nan
    jobs[phase].append((n or 0, tag, est))


# ---- phase 0: checks (one per setup family, N 128, compare with the existing 2080 Ti file) + probes
# E_probe_* / E_check_* exist from 2026-11-05 (step0_probe memory 256-512; h200check_z0.1 reruns)
for t in ("E_check_sedov", "E_check_cwb", "E_probe_sedov", "E_probe_cwb"):
    jobs[0].append((0, t, np.nan))
job(0, "H_check_wind", ["python wind_study.py --n 128 --name h200check_lam0.1 --kappa "
                        f"{plan['wind_lam0.1']['kappa']} --vred {plan['wind_lam0.1']['vred']}"],
    "wind/N128/lam0.1.npz", 128, 128)
p = plan["cp_lam0.1"]
job(0, "H_check_cp_lam", [f"python sedov_study.py --setup cp --n 128 --zeta 0.1 --kappa {p['kappa']} --vred {p['vred']} "
                          f"--name h200check_z0.1_lam0.1 --stop-radius {p['stop_radius']}"],
    "sedov/cp_p0.6/N128/z0.1_lam0.1.npz", 128, 128)

# ---- phase 1: hydro / advective ladders (same flags as the original study)
SED = {"pe": (256, 384, 448), "ss": (256, 384), "cp": (256, 384, 448)}
for setup, ns in SED.items():
    sdir = "cp_p0.6" if setup == "cp" else setup
    for n in ns:
        job(1, f"H_{setup}_control_N{n}",
            [f"python sedov_study.py --setup {setup} --n {n} --snapshots 30 --keep 9 14 19 24 29 --zeta 0"],
            f"sedov/{sdir}/N224/control.npz", n, 224)
        job(1, f"H_{setup}_z0.1_N{n}", [f"python sedov_study.py --setup {setup} --n {n} --zeta 0.1 --snapshots 30 --keep 29"],
            f"sedov/{sdir}/N224/z0.1.npz", n, 224)
        if setup == "cp":
            job(1, f"H_cp_kr13_N{n}", [f"python sedov_study.py --setup cp --n {n} --kr13 --snapshots 30 --keep 29"],
                "sedov/cp_p0.6/N224/kr13.npz", n, 224)
for n in (256, 320, 384):
    job(1, f"H_cwb_control_N{n}", [f"python cwb_study.py --n {n} --zeta 0"], "cwb/N192/control.npz", n, 192)
    job(1, f"H_cwb_z0.1_N{n}", [f"python cwb_study.py --n {n} --zeta 0.1"], "cwb/N192/z0.1.npz", n, 192)
for n in (256, 320):
    job(1, f"H_w_adv_N{n}", [f"python wind_study.py --n {n} --name adv --vred 0"], "wind/N192/adv.npz", n, 192)

# ---- phase 2: CR diffusion (D5)
for n in (256, 320):
    for lam in (0.03, 0.1, 0.3):
        q = plan[f"wind_lam{lam:g}"]
        job(2, f"H_w_lam{lam:g}_N{n}", [f"python wind_study.py --n {n} --name lam{lam:g} --kappa {q['kappa']} --vred {q['vred']}"],
            f"wind/N192/lam{lam:g}.npz", n, 192)
for n in (256, 384):
    for setup, models in (("cp", ("z0.1", "kr13")), ("ss", ("z0.1",))):
        sdir = "cp_p0.6" if setup == "cp" else setup
        for model in models:
            for lam in (0.03, 0.1):
                q = plan[f"{setup}_lam{lam:g}"]
                stop = f" --stop-radius {q['stop_radius']}" if q["stop_radius"] else ""
                job(2, f"H_{setup}_{model}_lam{lam:g}_N{n}",
                    [f"python sedov_study.py --setup {setup} --n {n} " + ("--kr13" if model == "kr13" else "--zeta 0.1")
                     + f" --kappa {q['kappa']} --vred {q['vred']} --name {model}_lam{lam:g}{stop}"],
                    f"sedov/{sdir}/N192/{model}_lam{lam:g}.npz", n, 192)

# ---- phase 3: optional
for setup in ("pe", "cp"):
    sdir = "cp_p0.6" if setup == "cp" else setup
    job(3, f"H_{setup}_z0.1_N512", [f"python sedov_study.py --setup {setup} --n 512 --zeta 0.1 --snapshots 30 --keep 29"],
        f"sedov/{sdir}/N224/z0.1.npz", 512, 224)
for n in (192, 256):
    job(3, f"H_cwb_kr13_N{n}", [f"python cwb_study.py --n {n} --kr13"], "cwb/N192/z0.1.npz", n, 192)

lines = ["H200 queue (launcher/mk_h200_jobs.py): est. = 2080 Ti hours (N^4 from the finest existing run);",
         "divide by the H200 speed-up measured in phase 0. Peak memory ~ 6.8 GB x (N/192)^3 (2080 Ti, peak_bytes_in_use).",
         ""]
for ph in (0, 1, 2, 3):
    order = sorted(jobs[ph], key=lambda j: (j[0], j[1])) if ph else jobs[ph]
    (RAW / "jobs" / f"queue_h200_{ph}.txt").write_text("\n".join(t for _, t, _ in order) + "\n")
    tot = np.nansum([e for *_, e in order])
    lines.append(f"phase {ph}: {len(order)} jobs, {tot:.0f} h (2080 Ti) -> queue_h200_{ph}.txt")
    for n, t, e in order:
        mem = 6.8 * (n / 192) ** 3 if n else np.nan
        lines.append(f"  {t:28s} {'' if np.isnan(e) else f'{e:6.1f} h'}  {'' if np.isnan(mem) else f'~{mem:4.0f} GB'}")
text = "\n".join(lines)
print(text)
(HERE / "pics" / "H200_queue_estimate.txt").write_text(text + "\n")
