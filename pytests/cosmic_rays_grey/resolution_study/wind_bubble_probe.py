"""Diagnostic for the cr_wind_bubble re-baseline: E_cr vs the injected energy (accumulator)
of the DSA run, with optional CosmicRayGreyConfig overrides. GPU only.

Usage: [MDOT_SCALE=0.67] python wind_bubble_probe.py [key=value ...]   e.g. dsa_continuous_normal_surface=0
"""

import sys

import common
from common import InjectionAccumulator, require_pinned_gpu

import os

import cr_wind_bubble as m

m.M_DOT = m.M_DOT * float(os.environ.get("MDOT_SCALE", "1"))  # 0.67: the pre-fix effective wind

overrides = {}
for item in sys.argv[1:]:
    key, value = item.split("=")
    overrides[key] = {"0": False, "1": True}.get(value, value)
base_cls = m.CosmicRayGreyConfig
m.CosmicRayGreyConfig = lambda **k: base_cls(**{**k, **overrides})

print("GPU", require_pinned_gpu(), "overrides", overrides, "MDOT_SCALE", os.environ.get("MDOT_SCALE", "1"), flush=True)
InjectionAccumulator.install()
acc = InjectionAccumulator().activate()
run = m._run_wind_bubble(m.DSA_EFFICIENCY)
cu = m.CODE_UNITS
L = 0.5 * m.M_DOT.to(cu.code_mass / cu.code_time).value * m.V_INF.to(cu.code_velocity).value ** 2
inj = acc.total()
print(f"E_cr {run['E_cr']:.5e}, injected {inj:.5e}, E_cr / injected {run['E_cr'] / inj:.3f}, "
      f"injected / (L t_end) {inj / (L * m.T_END):.4f}, E_cr / E_total {run['E_cr'] / run['E_total']:.4f}", flush=True)
