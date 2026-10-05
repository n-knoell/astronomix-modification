"""
SILCC-ISM project M7 (pilot): a Girichidis et al. (2016)-scale stratified box,
to reproduce the structure of their Fig. 1 (``pics/girichidis_SN.jpeg``) --
a dense, thin midplane layer after 250 Myr, with a CR-supported extended
atmosphere when CRs are injected.

**Diffusion convention changed 2026-10-04 (DESIGN.md "Open: CR diffusion correctness",
fix-plan step 1).** ``diffusion_coefficient`` is now the ``e_cr`` diffusivity, so the kappa
below now diffuses with ``D = kappa``. Every run made before then had ``D = kappa/3`` and a
3x faster relaxation rate ``nu``. Since fix-plan step 2 (same day) the relaxation is implicit,
so the dt_relax bound the cost reasoning below relies on (``dt <= C_cfl * kappa / v_red^2``) no
longer exists, and an isotropic ``kappa_perp = 1e26`` costs no extra steps. Since step 3, the
MHD "projection onto B" below is a per-stage tensor relaxation, and the paper's
``kappa_perp = 1e26`` can be set (``CosmicRayGreyParams.perpendicular_diffusion_coefficient``).
The run has not been redone yet (fix-plan step 5).

**NOT a committed pytest -- an exploratory script, not yet run.** M5's box
(37.5 x 37.5 x 300 pc) is narrower than one scale height and 26x over-driven,
so it cannot form that layer (SN bubbles cannot vent sideways and lift the
whole column instead). This script switches to the paper's own setup where
the code allows it, at half the paper's resolution:

| | Girichidis et al. 2016 (Sec. 2) | this pilot |
|---|---|---|
| box | 2 x 2 x +-20 kpc, AMR | 2 x 2 x +-2.5 kpc, uniform grid |
| cell size | 15.6 pc for abs(z) < 2.5 kpc | 15.6 pc (default ``--res=2``; 1.625 -> 19.2 pc, 0.5 -> 62.5 pc) |
| gas | 10 Msun/pc^2, scale height 60 pc, pressure equilibrium | same (Gaussian, hydrostatic in the potential below) |
| potential | Kuijken & Gilmore (1989) | same form, Joung & Mac Low (2006) constants |
| SN rate | 60 /Myr/kpc^2 (240 /Myr in the box) | same |
| SN heights | 80% type II (50 pc), 20% Ia (325 pc) | same (new ``gaussian_z_placement``) |
| SN energy | 1e51 erg thermal and/or 1e50 erg CR | same, three modes (CLI argument) |
| unresolved SNe | SILCC scheme (Gatto et al. 2015) | Kim & Ostriker (2015) momentum where unresolved (new ``momentum_injection_hybrid``) |
| magnetic field | B_x = 3 muG sqrt(rho / rho_0) (SILCC, Walch et al. 2015) | same (``--no-mhd``: hydro) |
| CR transport | kappa_par = 1e28, kappa_perp = 1e26 cm^2/s along B (MHD) | kappa_par = 1e28 along B, kappa_perp = 0 (F_cr projected onto B; isotropic 1e28 with ``--no-mhd``) |
| chemistry | H+/H/H2/C+/CO network with shielding | Koyama & Inutsuka (2002) cooling |
| self-gravity | yes (tree) | yes: FFT, periodic x/y, isolated z, + Jeans pressure floor (N_J = 4) (``--no-self-gravity``, ``--no-jeans-floor``) |
| t_end | 250 Myr | 250 Myr |

Usage::

    python m7_girichidis_pilot.py [both|thermal|cr] [--setup-only] [--res=F] [--t-end-myr=T] [--snapshots=N] [--sn-radius-pc=R] [--out-dir=DIR]
        [--no-mhd] [--no-self-gravity] [--no-jeans-floor] [--mhd-tolerance=double|single]
        [--half-height-kpc=H]

``both`` (default) is their "thermal + CR" run (1e51 erg thermal + 1e50 erg CR
per SN), ``thermal`` their thermal-only run (no CR transport at all, as in the
paper), ``cr`` their CR-only run (1e50 erg CR, no thermal energy).
``--setup-only`` builds the initial condition, prints the derived numbers and
the cost estimate, and exits without running. ``--res`` scales the grid
(2 = 15.6 pc, the default; 1.625 = 19.2 pc; 1 = 31.25 pc; 0.5 = 62.5 pc), ``--t-end-myr`` and
``--snapshots`` override the end time and snapshot count (for quick looks);
``--sn-radius-pc`` the SN injection radius (default 40 pc, fixed in pc). Every
SN is logged (time, height, local n_H, thermal or momentum mode) to
``pics/m7_pilot_sn_log_*.npz`` and summarized in ``pics/m7_pilot_sn_modes_*.svg``.
``--out-dir`` (default ``/export/scratch/nknoell``, created before the run starts) gets
the raw results right after the run, before any plotting:
``m7_pilot_data_*.npz`` with the final full state, density / pressure / v_z
(/ E_CR) of every snapshot as float32, the snapshot times and the SN log.

Design decisions (2026-10-03, discussed with the user; see the comparison in
PROGRESS.md):

1. **Resolution: the paper's 15.6 pc by default (``--res=2``, 128 x 128 x 320;
   the default was 1.625 = 19.2 pc until 2026-10-04).** Measured on an RTX 2080 Ti (2026-10-03, ``both``, first 2 Myr,
   compilation excluded): 2.47 / 287 / 566 s per simulated Myr at
   ``--res`` = 0.5 / 1.5 / 1.75, i.e. ~res^4.33 -- steeper than cells x steps
   (res^4), since the 2-cell SN footprint holds less gas at finer grids, so
   remnants get hotter and the CFL dt smaller. The full 250 Myr at 0.5 took
   1177 s, 1.8x its early rate (the gas heats up later). That gives ``both``
   at 17.9 pc ~39-71 h on a 2080 Ti, ~5-18 h on an H200 at an assumed 8-4x
   speed-up (not measured); the paper's 15.6 pc (``--res=2``) is ~1.8x more.
   Memory on the 11 GB 2080 Ti: ``--res`` <= 1.75 fits (8.3 GB at 1.75 with 2
   snapshots, plus ~0.25 GB per stored snapshot, so ``--snapshots`` <= ~9
   there); 2 and 2.5 run out of memory. ``--res`` needs 64 x res to be an
   integer (e.g. 1.1875 = 26.3 pc, 1.25 = 25 pc). There is no checkpointing
   (all snapshots come back at the end), so a run killed at a wall-time limit
   loses everything: measure first with ``--t-end-myr=10``.
2. **CR transport: isotropic kappa = 1e28 cm^2/s, reduced streaming speed
   1000 km/s.** The explicit F_cr relaxation bounds
   ``dt <= C_cfl * kappa / v_red^2``; with v_red = 1000 km/s this is ~13 kyr,
   matching the hydro CFL at 31.25 pc. v_red = 3000 km/s (safely above the
   hottest gas) would cost ~9x more steps *and* worsen DESIGN.md's open
   Finding 1 (v_red inflates the shared HLL wave-speed bound and smears
   shocks/phase boundaries -- exactly the thin cold layer this run is
   looking for). Accepted trade-off: the hottest remnants briefly exceed
   v_red. kappa_perp = 1e26 isotropically (M5's choice) would make dt_relax
   100x smaller and is unaffordable here.
3. **Several SNe per step** (new ``SNDrivingConfig.max_sn_per_step``): at
   240 SNe/Myr and dt ~ 12 kyr, ~3 SNe go off per step; the original
   one-trial-per-step driver would silently cap the rate at 1/dt.
4. **Unresolved SNe get momentum** (new ``momentum_injection_hybrid``): at
   31 pc a 1e51 erg deposit in the disc heats its ~6e4 Msun footprint to only
   ~1e5 K, at the peak of the cooling curve -- pure thermal injection
   overcools (M4's finding, much worse here). Per SN, the injection radius
   is compared with the Kim & Ostriker (2015) shell-formation radius at the
   local density: thermal if resolved (tenuous gas, mostly type Ia heights),
   terminal momentum + a 1% thermal floor otherwise. Delayed cooling (M4/M5)
   was not used: shielding long enough for a remnant to expand over a
   62 pc footprint (~1 Myr) would keep ~half the midplane layer from cooling
   at this SN rate, suppressing the very layer we want to see.
   **The injection radius is fixed at 40 pc** (``--sn-radius-pc``; it was 2
   cells until 2026-10-04): the switch threshold n_H ~ (22.6 pc / radius)^2.38
   = 0.26 cm^-3 then no longer moves with ``--res``. With 2 cells it was
   0.46 cm^-3 at 15.6 pc, and the 15.6 pc run's midplane density sat at
   0.44-0.47 from ~100 Myr on, i.e. right at the switch. Every SN's mode is
   logged (``SNDrivingConfig.log_sn_events``) to check this.
5. **MHD and self-gravity on by default (2026-10-04)**, as in the paper;
   ``--no-mhd`` / ``--no-self-gravity`` give the earlier hydro / external-
   potential-only setup.
   - MHD: SILCC's initial field B_x = 3 muG sqrt(rho / rho_0) (Walch et al.
     2015; Girichidis et al. 2016 only say "ordered"), FV MHD (Strang-split
     gas half-steps around the implicit Pang & Wu B/v update). CR transport
     becomes anisotropic: F_cr is projected onto B once per step, i.e.
     kappa_par = 1e28 and kappa_perp = 0 (paper: 1e26). Unlike an isotropic
     kappa_perp = 1e26 this costs no extra steps (dt_relax uses kappa_par).
     The CFL estimate includes the Alfven speed (new opt-in
     ``SimulationConfig.fv_mhd_alfven_cfl``; v_A reached ~400 km/s in the
     31 pc pilot, and in ``thermal`` mode no v_red bounds dt); the
     per-snapshot summary prints max v_A. The
     B/v fixed-point tolerance is 1e-10 (``--mhd-tolerance=double``,
     needed for energy closure, cr_mhd_energy_budget.py) or 1e-5.
   - Self-gravity: FFT Poisson solve, periodic in x/y and isolated in z
     (``_poisson_periodic_xy_isolated_z``, validated in
     ``pytests/self_gravity/mixed_boundary_poisson.py``), added to the
     Kuijken & Gilmore potential. With no sink particles and 15-60 pc cells,
     cold gas would collapse into single cells, so a Truelove Jeans pressure
     floor (Jeans length >= 4 cells, ``CoolingConfig.jeans_pressure_floor``)
     is on with self-gravity (``--no-jeans-floor`` to switch it off).
   - The initial condition is hydrostatic in external + gas self-gravity,
     with thermal + magnetic pressure (magnetic pressure B^2/8pi ~ 0.5x the
     thermal pressure at the midplane).

Open risks (see PROGRESS.md / DESIGN.md): 250 Myr is ~30x longer than any
stratified-box run so far; the hybrid switch and multi-SN driver are new
(scratch-checked, not yet exercised in a long run); Kuijken & Gilmore
constants are taken from Joung & Mac Low (2006), not re-checked against
Walch et al. (2015).
"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1, interval=1)
# ruff: noqa: E402
# =======================

# general
import sys
import time
from pathlib import Path

# jax
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# numerics
import numpy as np

# units
from astropy import units as u
import astropy.constants as c

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# astronomix containers
from astronomix.option_classes.simulation_config import SnapshotSettings
from astronomix import CARTESIAN, FINITE_VOLUME, HLLC, MINMOD
from astronomix import OPEN_BOUNDARY, PERIODIC_BOUNDARY
from astronomix import BoundarySettings, BoundarySettings1D, GravityConfig, CodeUnits
from astronomix import SimulationConfig, SimulationParams
from astronomix.option_classes.simulation_config import StaticFloatVector, StaticIntVector
from astronomix.option_classes.simulation_config import DOUBLE_PRECISION, SINGLE_PRECISION

# astronomix functions
from astronomix import (
    construct_primitive_state,
    finalize_config,
    get_helper_data,
    get_registered_variables,
    time_integration,
)

# astronomix modules
from astronomix._modules._cooling._cooling import (
    get_effective_molecular_weights,
    get_pressure_from_temperature,
    get_temperature_from_pressure,
)
from astronomix._modules._cooling._cooling_tables import koyama_inutsuka_cooling
from astronomix._modules._cooling.cooling_options import (
    IMPLICIT_COOLING,
    KOYAMA_INUTSUKA_NET_COOLING,
    CoolingConfig,
    CoolingCurveConfig,
    CoolingParams,
)
from astronomix._modules._sn_driving.sn_driving_options import SNDrivingConfig, SNDrivingParams
from astronomix._modules._sn_driving.sn_driving import SN_EVENT_LOG
from astronomix._modules._cosmic_rays_grey.cosmic_ray_grey_options import (
    CosmicRayGreyConfig,
    CosmicRayGreyParams,
)

# ---- run mode (CLI) ----
_ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
MODE = _ARGS[0] if _ARGS else "both"
SETUP_ONLY = "--setup-only" in sys.argv
# optional overrides for quick provisional runs, e.g. --res=0.5 --t-end-myr=50 --snapshots=11
_OPTS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
if MODE not in ("both", "thermal", "cr"):
    raise SystemExit(f"unknown mode {MODE!r}; use both, thermal or cr")
CR_ACTIVE = MODE != "thermal"  # thermal-only: no CR transport at all, as in the paper
MHD = "--no-mhd" not in sys.argv
SELF_GRAVITY = "--no-self-gravity" not in sys.argv
JEANS_FLOOR = SELF_GRAVITY and "--no-jeans-floor" not in sys.argv
PHYSICS_LABEL = ", ".join(["MHD" if MHD else "hydro"]
                          + (["self-gravity" + ("" if JEANS_FLOOR else " (no Jeans floor)")] if SELF_GRAVITY else []))

# ---- units and gas (same as M5) ----
GAMMA = 5.0 / 3.0
GAMMA_CR = 4.0 / 3.0
X_H = 0.76
Z_METAL = 0.02
MU, MU_E, MU_H = get_effective_molecular_weights(X_H, Z_METAL)
CODE_UNITS = CodeUnits(1 * u.pc, 1 * u.M_sun, 1 * u.km / u.s)
KI_PARAMS = koyama_inutsuka_cooling(CODE_UNITS, X_H, Z_METAL)
FLOOR_TEMPERATURE_CODE = 10.0 * KI_PARAMS.code_temperature_per_kelvin


def _code(quantity, unit):
    return float(quantity.to(unit).value)


CODE_TIME = CODE_UNITS.code_time
CODE_DENSITY = CODE_UNITS.code_density
MYR_CODE = _code(1 * u.Myr, CODE_TIME)

# ---- box and grid ----
# _RESOLUTION_FACTOR = 2 (default): 128 x 128 x 320 cells, dx = 15.6 pc (Girichidis'
# own, the H200 run, see decision 1); 1.75: 17.9 pc; 1.625: 19.2 pc;
# 1: 64 x 64 x 160, 31.25 pc; 0.5: 32 x 32 x 80, 62.5 pc (~20 min on a 2080 Ti).
_RESOLUTION_FACTOR = float(_OPTS.get("res", 2.0))
L_XY = 2000.0  # pc
# +-2.5 kpc around the midplane by default (the paper's uniform-resolution part);
# --half-height-kpc enlarges it (open top, CR diffusion length over 250 Myr 2.9 kpc)
HALF_HEIGHT_KPC = float(_OPTS.get("half-height-kpc", 2.5))
L_Z = 2000.0 * HALF_HEIGHT_KPC  # pc
Z0 = 0.5 * L_Z
N_XY = int(round(64 * _RESOLUTION_FACTOR))
N_Z = int(round(N_XY * L_Z / L_XY))
DX = L_XY / N_XY
assert abs(L_Z / N_Z - DX) < 1e-12
NUM_SNAPSHOTS = int(_OPTS.get("snapshots", 26))  # default: every 10 Myr
T_END = float(_OPTS.get("t-end-myr", 250.0)) * MYR_CODE
OUT_SUFFIX = (f"_{MODE}{'_mhd' if MHD else ''}{'_sg' if SELF_GRAVITY else ''}"
              f"{'_nojeans' if SELF_GRAVITY and not JEANS_FLOOR else ''}"
              f"{f'_z{HALF_HEIGHT_KPC:g}kpc' if HALF_HEIGHT_KPC != 2.5 else ''}_{N_Z}")
OUT_DIR = Path(_OPTS.get("out-dir", "/export/scratch/nknoell"))

# ---- disc: Sigma = 10 Msun/pc^2, Gaussian with scale height 60 pc ----
SIGMA_GAS = 10.0  # Msun / pc^2 (code surface density)
H_GAS = 60.0  # pc, Gaussian standard deviation (Girichidis' "scale height")
RHO_MIDPLANE = SIGMA_GAS / (np.sqrt(2.0 * np.pi) * H_GAS)  # Msun / pc^3 = code density
RHO_BACKGROUND = _code(1e-27 * u.g / u.cm**3, CODE_DENSITY)  # bottom of their colour scale
T_BACKGROUND_KELVIN = 1.0e6  # sets the pressure at the box top (hot halo)
MIN_DENSITY_CODE = _code(1e-28 * u.g / u.cm**3, CODE_DENSITY)
MIN_PRESSURE_CODE = float(get_pressure_from_temperature(MIN_DENSITY_CODE, FLOOR_TEMPERATURE_CODE, X_H, Z_METAL))
N_H_PER_CODE_DENSITY = _code(1 * CODE_DENSITY, u.g / u.cm**3) / (MU_H * c.m_p.to(u.g).value)

# ---- Kuijken & Gilmore (1989) potential, Joung & Mac Low (2006) eq. 4 constants ----
# g_z = -a1 z / sqrt(z^2 + z0^2) - a2 z  ->  phi = a1 (sqrt(z^2 + z0^2) - z0) + a2 z^2 / 2
KG_A1 = _code(1.42e-3 * u.kpc / u.Myr**2, CODE_UNITS.code_length / CODE_TIME**2)
KG_A2 = _code(5.49e-4 / u.Myr**2, 1 / CODE_TIME**2)
KG_Z0 = 180.0  # pc

# ---- supernovae ----
SN_RATE_PER_KPC2_MYR = 60.0
SN_RATE_CODE = SN_RATE_PER_KPC2_MYR * (L_XY / 1000.0) ** 2 / MYR_CODE
# dt cap max_sn_per_step / sn_rate = 16.7 kyr, above the dt at --res >= 1 (~12 kyr at
# 31 pc, smaller at finer grids); the expected SN rate is exact for any value. Was 16
# until 2026-10-04: every trial costs grid passes, 68 vs 45 s/Myr against 2 at --res=1.
MAX_SN_PER_STEP = 4
E_THERMAL_ERG = {"both": 1e51, "thermal": 1e51, "cr": 0.0}[MODE]
E_CR_ERG = {"both": 1e50, "thermal": 0.0, "cr": 1e50}[MODE]
SN_ENERGY_CODE = _code((E_THERMAL_ERG + E_CR_ERG) * u.erg, CODE_UNITS.code_energy)
SN_CR_FRACTION = E_CR_ERG / (E_THERMAL_ERG + E_CR_ERG)
# Fixed in pc (2026-10-04; was 2 cells). With the hybrid switch, an SN is
# deposited thermally only below n_H ~ (r_sf(n=1) / radius)^(1/0.42), so a
# radius tied to the grid moved that threshold with resolution (0.46 cm^-3 at
# 15.6 pc, 0.13 at 26.3 pc) and the --res=2 run's midplane density sat right
# at it. 40 pc puts the threshold at 0.26 cm^-3 for every --res; it is 2.6 / 2.1
# / 1.5 cells at --res = 2 / 1.625 / 1.1875 (warning below 1.5 cells).
SN_INJECTION_RADIUS = float(_OPTS.get("sn-radius-pc", 40.0))
SN_SMOOTH_CELLS = 1.0
SN_TYPE_II_FRACTION, SN_TYPE_II_HEIGHT, SN_TYPE_IA_HEIGHT = 0.8, 50.0, 325.0  # pc
# Kim & Ostriker (2015): terminal momentum 2.8e5 Msun km/s (code momentum
# units), shell-formation radius 22.6 pc * E51^0.29 * n0^-0.42; scaled with
# the thermal part of the SN energy (zero in CR-only mode -> no momentum).
E51 = E_THERMAL_ERG / 1e51
SN_MOMENTUM_COEFFICIENT = 2.8e5 * E51
SN_SHELL_FORMATION_RADIUS_COEFFICIENT = 22.6 * E51**0.29 if E51 > 0 else 0.0
SN_MOMENTUM_THERMAL_FLOOR = 0.01  # M4's value (numerical floor, see SNDrivingParams)
SN_MOMENTUM_DENSITY_REFERENCE = 1.0 / N_H_PER_CODE_DENSITY  # code density of n_H = 1 cm^-3
M_STAR_PER_SN = 100.0  # Msun, Girichidis' "one massive star per 100 Msun"
SFR_CODE = SN_RATE_CODE * M_STAR_PER_SN  # Msun per code time

# ---- cosmic rays ----
KAPPA_CODE = _code(1e28 * u.cm**2 / u.s, CODE_UNITS.code_length**2 / CODE_TIME)
V_RED_CODE = 1000.0  # km/s = code velocity

# ---- magnetic field and self-gravity ----
# SILCC (Walch et al. 2015): B_x = 3 muG sqrt(rho / rho_0). Code units: magnetic
# pressure is B^2 / 2, so B_code = sqrt(2 P_B) with P_B = B^2 / (8 pi) in code
# pressure units (3 muG ~ 1.03 code).
B0_GAUSS = 3e-6
CODE_PRESSURE_ERG_CM3 = _code(1 * CODE_UNITS.code_energy / CODE_UNITS.code_length**3, u.erg / u.cm**3)
P_B_MIDPLANE_CODE = B0_GAUSS**2 / (8 * np.pi) / CODE_PRESSURE_ERG_CM3
G_CODE = _code(c.G, CODE_UNITS.code_length * (CODE_UNITS.code_length / CODE_TIME) ** 2 / CODE_UNITS.code_mass)
JEANS_FLOOR_CELLS = 4.0
MHD_TOLERANCE = _OPTS.get("mhd-tolerance", "double")

# ---- diagnostics ----
Z_OUTFLOW = 1000.0  # pc, Girichidis' reference height
EDGE_ON_HALF_HEIGHT = 1500.0  # pc, their Fig. 1 edge-on panels
DENSITY_RANGE = (1e-27, 1e-21)  # g/cm^3, their colour bar
E_CR_RANGE = (1e-12, 1e-10)  # erg/cm^3, their colour bar
DENSITY_CMAP, E_CR_CMAP = "jet", "bwr"  # match their figure for side-by-side comparison

# ---- cost model (see module docstring, decision 1) ----
# measured (RTX 2080 Ti, 2026-10-03, ``both``): wall time per simulated Myr over the
# first 2 Myr, compilation excluded -- 2.47 s at --res=0.5, 287 s at 1.5, 566 s at
# 1.75, i.e. ~ res^4.33 (cells x steps, plus smaller, hotter SN footprints at finer
# grids). The full 250 Myr at --res=0.5 averaged 1.8x its early rate.
EARLY_S_PER_MYR_REF, REF_RES, RES_EXPONENT = 2.47, 0.5, 4.33
LATE_FACTOR_RANGE = (1.0, 1.8)
MODE_FACTOR = {"both": 1.0, "cr": 1.0, "thermal": 0.67}[MODE]  # thermal/both = 785/1177 s at --res=0.5
H200_SPEEDUPS = (4.0, 8.0)  # assumed range vs. a 2080 Ti; not measured for this code
V_HOT_ESTIMATE = 2600.0  # km/s, sound speed at the ~3e8 K peak of the provisional runs


def _base_config() -> SimulationConfig:
    return SimulationConfig(
        geometry=CARTESIAN,
        solver_mode=FINITE_VOLUME,
        riemann_solver=HLLC,
        limiter=MINMOD,
        dimensionality=3,
        box_size=StaticFloatVector(L_XY, L_XY, L_Z),
        num_cells=StaticIntVector(N_XY, N_XY, N_Z),
        exact_end_time=True,
        boundary_settings=BoundarySettings(
            x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
            z=BoundarySettings1D(OPEN_BOUNDARY, OPEN_BOUNDARY),
        ),
        mhd=MHD,
        fv_mhd_alfven_cfl=MHD,
        numerical_precision={"double": DOUBLE_PRECISION, "single": SINGLE_PRECISION}[MHD_TOLERANCE],
        gravity_config=GravityConfig(external_potential=True, self_gravity=SELF_GRAVITY),
        cooling_config=CoolingConfig(
            cooling=True,
            cooling_method=IMPLICIT_COOLING,
            cooling_curve_config=CoolingCurveConfig(cooling_curve_type=KOYAMA_INUTSUKA_NET_COOLING),
            subcycle_stiff_cooling=True,
            jeans_pressure_floor=JEANS_FLOOR,
        ),
        sn_driving_config=SNDrivingConfig(
            sn_driving=True,
            gaussian_z_placement=True,
            max_sn_per_step=MAX_SN_PER_STEP,
            momentum_injection=E51 > 0,
            momentum_injection_hybrid=E51 > 0,
            log_sn_events=True,
        ),
        cosmic_ray_grey_config=CosmicRayGreyConfig(
            grey_cosmic_rays=CR_ACTIVE, diffusive_relaxation=CR_ACTIVE,
            anisotropic_transport=CR_ACTIVE and MHD,
        ),
        progress_bar=True,
        return_snapshots=True,
        num_snapshots=NUM_SNAPSHOTS,
        snapshot_settings=SnapshotSettings(
            return_states=True,
            return_final_state=True,
            return_total_mass=True,
        ),
    )


def _kg_potential(z_rel):
    return KG_A1 * (np.sqrt(z_rel**2 + KG_Z0**2) - KG_Z0) + 0.5 * KG_A2 * z_rel**2


def _kg_gradient(z_rel):
    return KG_A1 * z_rel / np.sqrt(z_rel**2 + KG_Z0**2) + KG_A2 * z_rel


def _density_profile(z_rel):
    return np.maximum(RHO_MIDPLANE * np.exp(-0.5 * (z_rel / H_GAS) ** 2), RHO_BACKGROUND)


def _magnetic_pressure(z_rel):
    """B^2 / 2 (code) of B_x = B0 sqrt(rho / rho_0): proportional to rho."""
    return P_B_MIDPLANE_CODE * _density_profile(z_rel) / RHO_MIDPLANE if MHD else 0.0 * z_rel


def _hydrostatic_pressure(z_rel_cells):
    """Thermal pressure P(z) of hydrostatic equilibrium d(P + P_B)/dz = -rho
    g, g = Kuijken & Gilmore + (with self-gravity) the gas layer's own slab
    field 2 pi G Sigma(-z..z), integrated on a fine grid from the box top (thermal pressure of
    RHO_BACKGROUND at T_BACKGROUND_KELVIN, plus its magnetic pressure) down
    to the midplane, mirrored below it, interpolated to the cell centres; the
    magnetic pressure is then subtracted."""
    z_fine = np.linspace(0.0, 0.5 * L_Z + DX, 200001)
    rho_fine = _density_profile(z_fine)
    g_fine = _kg_gradient(z_fine)
    if SELF_GRAVITY:
        column = np.concatenate([[0.0], np.cumsum(0.5 * (rho_fine[1:] + rho_fine[:-1]) * np.diff(z_fine))])
        g_fine = g_fine + 2 * np.pi * G_CODE * 2 * column
    integrand = rho_fine * g_fine
    # integral from z to the top: total minus cumulative from 0
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(z_fine))])
    above = cumulative[-1] - cumulative
    p_top = float(get_pressure_from_temperature(
        RHO_BACKGROUND, T_BACKGROUND_KELVIN * KI_PARAMS.code_temperature_per_kelvin, X_H, Z_METAL,
    ))
    p_total = np.interp(np.abs(z_rel_cells), z_fine, p_top + _magnetic_pressure(z_fine[-1]) + above)
    p_thermal = p_total - _magnetic_pressure(z_rel_cells)
    assert np.all(p_thermal > 0), "magnetic pressure exceeds the hydrostatic total pressure"
    return p_thermal


def _potential_and_ic(config, registered_variables):
    helper_data = get_helper_data(config)
    z = np.asarray(helper_data.geometric_centers[..., 2])
    z_rel = z - Z0
    phi = jnp.asarray(_kg_potential(z_rel))
    rho = jnp.asarray(_density_profile(z_rel))
    p = jnp.asarray(_hydrostatic_pressure(z_rel))
    zero = jnp.zeros_like(rho)
    magnetic = {}
    if MHD:
        magnetic = dict(magnetic_field_x=jnp.sqrt(2.0 * jnp.asarray(_magnetic_pressure(z_rel))),
                        magnetic_field_y=zero, magnetic_field_z=zero)
    initial_state = construct_primitive_state(
        config=config, registered_variables=registered_variables,
        density=rho, velocity_x=zero, velocity_y=zero, velocity_z=zero, gas_pressure=p, **magnetic,
    )
    return phi, initial_state, helper_data


def _cost_estimate():
    """Wall-time estimate from the measured early-time cost per simulated Myr,
    scaled as res^4.33 (module docstring, decision 1)."""
    t_myr = T_END / MYR_CODE
    s_per_myr = EARLY_S_PER_MYR_REF * (_RESOLUTION_FACTOR / REF_RES) ** RES_EXPONENT * MODE_FACTOR
    lo, hi = (s_per_myr * t_myr * f / 3600.0 for f in LATE_FACTOR_RANGE)
    print(
        f"[cost] ~{s_per_myr:.0f} s per simulated Myr early on (RTX 2080 Ti) -> ~{lo:.1f}-{hi:.1f} h for "
        f"{t_myr:.0f} Myr on a 2080 Ti; ~{lo / max(H200_SPEEDUPS):.1f}-{hi / min(H200_SPEEDUPS):.1f} h on an "
        f"H200 at {min(H200_SPEEDUPS):.0f}-{max(H200_SPEEDUPS):.0f}x (not measured). "
        f"Mean SN rate x dt cap: max_sn_per_step = {MAX_SN_PER_STEP}."
        + (" Measured for hydro without self-gravity: MHD (two gas half-steps + the B/v"
           " fixed point) and the Poisson solves add to this." if MHD or SELF_GRAVITY else "")
    )


def _mass_heights(rho, z_rel):
    """Heights enclosing 70% and 90% of the gas mass (Girichidis' abstract
    quotes ~0.2 / ~1.5 kpc for the CR runs)."""
    column = rho.sum(axis=(0, 1))
    order = np.argsort(np.abs(z_rel))
    cumulative = np.cumsum(column[order]) / column.sum()
    abs_sorted = np.abs(z_rel)[order]
    return float(np.interp(0.7, cumulative, abs_sorted)), float(np.interp(0.9, cumulative, abs_sorted))


def _outflow(state, registered_variables, z_rel):
    """Mass outflow through both planes abs(z) = Z_OUTFLOW (positive = away
    from the midplane), flux-weighted outflow speed, mass-loading factor."""
    rho = np.asarray(state[registered_variables.density_index])
    vz = np.asarray(state[registered_variables.velocity_index.z])
    i_up = int(np.argmin(np.abs(z_rel - Z_OUTFLOW)))
    i_dn = int(np.argmin(np.abs(z_rel + Z_OUTFLOW)))
    flux_up = np.where(vz[:, :, i_up] > 0, rho[:, :, i_up] * vz[:, :, i_up], 0.0)
    flux_dn = np.where(vz[:, :, i_dn] < 0, -rho[:, :, i_dn] * vz[:, :, i_dn], 0.0)
    mdot = float((flux_up.sum() + flux_dn.sum()) * DX**2)
    mass = np.where(vz[:, :, i_up] > 0, rho[:, :, i_up], 0.0).sum() + np.where(vz[:, :, i_dn] < 0, rho[:, :, i_dn], 0.0).sum()
    v_out = float((flux_up.sum() + flux_dn.sum()) / mass) if mass > 0 else 0.0
    return mdot, v_out, mdot / SFR_CODE


def _scale_height(profile, z_rel, mid):
    target = profile[mid] / np.e
    upper = profile[mid:]
    below = np.nonzero(upper <= target)[0]
    if profile[mid] <= 0 or below.size == 0:
        return float("nan")
    i = below[0]
    if i == 0:
        return 0.0
    pa, pb, za, zb = upper[i - 1], upper[i], z_rel[mid + i - 1], z_rel[mid + i]
    return float(za + (np.log(pa) - np.log(target)) / (np.log(pa) - np.log(pb)) * (zb - za))


def _structure_plot(state, registered_variables, mid, t_myr, out_path):
    """Girichidis et al. (2016) Fig. 1 layout for one run: edge-on density
    (+-1.5 kpc), face-on midplane density, midplane CR energy density --
    their axis ranges, colour ranges and colour maps."""
    rho = np.asarray(state[registered_variables.density_index]) * _code(1 * CODE_DENSITY, u.g / u.cm**3)
    xy = [-L_XY / 2000.0, L_XY / 2000.0]
    n_panels = 3 if CR_ACTIVE else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 6.0))
    k = int(round(EDGE_ON_HALF_HEIGHT / DX))
    edge = rho[:, N_XY // 2, mid - k:mid + k].T
    im = axes[0].imshow(edge, origin="lower", extent=xy + [-EDGE_ON_HALF_HEIGHT / 1000.0, EDGE_ON_HALF_HEIGHT / 1000.0],
                        norm=LogNorm(*DENSITY_RANGE), cmap=DENSITY_CMAP, aspect="equal")
    axes[0].set(xlabel="x (kpc)", ylabel="z (kpc)", title="density, edge-on")
    fig.colorbar(im, ax=axes[0], label=r"density (g cm$^{-3}$)")
    im = axes[1].imshow(rho[:, :, mid].T, origin="lower", extent=xy + xy,
                        norm=LogNorm(*DENSITY_RANGE), cmap=DENSITY_CMAP, aspect="equal")
    axes[1].set(xlabel="x (kpc)", ylabel="y (kpc)", title="density, midplane")
    fig.colorbar(im, ax=axes[1], label=r"density (g cm$^{-3}$)")
    if CR_ACTIVE:
        e_unit = _code(1 * CODE_UNITS.code_energy / CODE_UNITS.code_length**3, u.erg / u.cm**3)
        e_cr = np.asarray(state[registered_variables.cosmic_ray_e_index])[:, :, mid] * e_unit
        im = axes[2].imshow(np.clip(e_cr, *E_CR_RANGE).T, origin="lower", extent=xy + xy,
                            norm=LogNorm(*E_CR_RANGE), cmap=E_CR_CMAP, aspect="equal")
        axes[2].set(xlabel="x (kpc)", ylabel="y (kpc)", title=r"$E_{\rm CR}$, midplane")
        fig.colorbar(im, ax=axes[2], label=r"$E_{\rm CR}$ (erg cm$^{-3}$)")
    label = {"both": "thermal + CR", "thermal": "only thermal", "cr": "only CR"}[MODE]
    fig.suptitle(f"M7 pilot ({PHYSICS_LABEL}), SNe: {label}  |  t = {t_myr:.1f} Myr  |  dx = {DX:.1f} pc "
                 f"(layout and colour scales of Girichidis et al. 2016 Fig. 1)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _edge_on_animation(states, registered_variables, mid, t_myr, frames, out_path, fps=4):
    """GIF of the edge-on density slice (y = box centre, abs(z) <= 1.5 kpc) over
    the given snapshot indices, with the same colour scale/map as
    ``_structure_plot``."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    to_cgs = _code(1 * CODE_DENSITY, u.g / u.cm**3)
    k = int(round(EDGE_ON_HALF_HEIGHT / DX))
    extent = [-L_XY / 2000.0, L_XY / 2000.0, -EDGE_ON_HALF_HEIGHT / 1000.0, EDGE_ON_HALF_HEIGHT / 1000.0]

    def edge(i):
        return np.asarray(states[i][registered_variables.density_index])[:, N_XY // 2, mid - k:mid + k].T * to_cgs

    fig, ax = plt.subplots(figsize=(4.6, 6.0))
    im = ax.imshow(edge(frames[0]), origin="lower", extent=extent, norm=LogNorm(*DENSITY_RANGE),
                   cmap=DENSITY_CMAP, aspect="equal")
    fig.colorbar(im, ax=ax, label=r"density (g cm$^{-3}$)")
    ax.set(xlabel="x (kpc)", ylabel="z (kpc)")
    title = ax.set_title("")
    label = {"both": "thermal + CR", "thermal": "only thermal", "cr": "only CR"}[MODE]

    def update(j):
        i = frames[j]
        im.set_data(edge(i))
        title.set_text(f"{label}, {PHYSICS_LABEL}\nt = {t_myr[i]:.0f} Myr, dx = {DX:.1f} pc")
        return im, title

    fig.tight_layout()
    FuncAnimation(fig, update, frames=len(frames), blit=False).save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def _sn_mode_report(t_myr, pics):
    """Save the SN event log and summarize the thermal/momentum split: count vs.
    the expected sn_rate * t_end, thermal fraction per 10 Myr, and the local
    n_H of each SN against the hybrid threshold."""
    if not SN_EVENT_LOG:
        print("[sn log] no SN events recorded")
        return
    log = np.array(SN_EVENT_LOG, dtype=float)
    t_ev, z_ev, n_ev, thermal = log[:, 0] / MYR_CODE, log[:, 1] - Z0, log[:, 2], log[:, 3].astype(bool)
    np.savez(pics / f"m7_pilot_sn_log{OUT_SUFFIX}.npz", t_myr=t_ev, z_pc=z_ev, n_h=n_ev, thermal=thermal)
    expected = SN_RATE_CODE * T_END
    n_threshold = (SN_SHELL_FORMATION_RADIUS_COEFFICIENT / SN_INJECTION_RADIUS) ** (1 / 0.42) if E51 > 0 else np.nan
    late = t_ev >= 0.75 * t_ev.max()
    print(f"[sn log] {len(t_ev)} SNe (expected {expected:.0f}); thermal (resolved) fraction "
          f"{thermal.mean():.3f} overall, {thermal[late].mean():.3f} in the last quarter; "
          f"threshold n_H = {n_threshold:.3g} cm^-3; median local n_H {np.median(n_ev):.3g} "
          f"(abs(z) < 100 pc: {np.median(n_ev[np.abs(z_ev) < 100]) if np.any(np.abs(z_ev) < 100) else np.nan:.3g})")

    # 10 Myr bins (shorter for short runs), the last one cut at t_end; rates
    # use each bin's actual width
    t_end_myr = T_END / MYR_CODE
    bin_width = 10.0 if t_end_myr >= 50.0 else t_end_myr / 10.0
    edges = np.append(np.arange(0.0, t_end_myr, bin_width), t_end_myr)
    idx = np.clip(np.digitize(t_ev, edges) - 1, 0, len(edges) - 2)
    centres = 0.5 * (edges[1:] + edges[:-1])
    counts = np.bincount(idx, minlength=len(centres))
    n_thermal = np.bincount(idx, weights=thermal, minlength=len(centres))
    with np.errstate(invalid="ignore", divide="ignore"):
        fraction = n_thermal / counts
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 4.6))
    ax0.plot(centres, fraction, "-o", ms=3)
    ax0.set(xlabel="t (Myr)", ylabel=f"thermal (resolved) fraction per {bin_width:g} Myr", ylim=(-0.02, 1.02),
            title="SN deposit mode vs. time")
    ax1.plot(centres, counts / np.diff(edges), "-o", ms=3)
    ax1.axhline(SN_RATE_CODE * MYR_CODE, color="0.5", ls="--", lw=1, label="nominal rate")
    ax1.set(xlabel="t (Myr)", ylabel="SNe per Myr", title="SN rate")
    ax1.legend()
    bins = np.logspace(np.log10(max(n_ev.min(), 1e-6)), np.log10(n_ev.max() * 1.01), 60)
    ax2.hist(n_ev[thermal], bins=bins, alpha=0.7, label="thermal")
    ax2.hist(n_ev[~thermal], bins=bins, alpha=0.7, label="momentum")
    if np.isfinite(n_threshold):
        ax2.axvline(n_threshold, color="k", ls="--", lw=1, label="hybrid threshold")
    ax2.set(xscale="log", xlabel=r"local ambient $n_H$ at the SN (cm$^{-3}$)", ylabel="SNe",
            title="local density of each SN")
    ax2.legend()
    fig.suptitle(f"M7 pilot ({MODE}, {PHYSICS_LABEL}), dx = {DX:.1f} pc, SN radius {SN_INJECTION_RADIUS:.0f} pc")
    fig.tight_layout()
    fig.savefig(pics / f"m7_pilot_sn_modes{OUT_SUFFIX}.svg")
    plt.close(fig)


def _save_run_data(snapshot_data, registered_variables, wall):
    """Write the raw results to OUT_DIR before any post-processing, so a
    crash in the plotting below does not lose the run."""
    rv = registered_variables
    fields = {"density": rv.density_index, "pressure": rv.pressure_index, "velocity_z": rv.velocity_index.z}
    if MHD:
        fields.update(magnetic_x=rv.magnetic_index.x, magnetic_y=rv.magnetic_index.y, magnetic_z=rv.magnetic_index.z)
    if CR_ACTIVE:
        fields["cosmic_ray_e"] = rv.cosmic_ray_e_index
    states = snapshot_data.states
    log = np.array(SN_EVENT_LOG, dtype=float).reshape(-1, 4)
    path = OUT_DIR / f"m7_pilot_data{OUT_SUFFIX}.npz"
    np.savez(
        path,
        time_myr=np.asarray(snapshot_data.time_points) / MYR_CODE,
        final_state=np.asarray(snapshot_data.final_state),
        **{f"snap_{name}": np.stack([np.asarray(states[i][j], dtype=np.float32) for i in range(len(states))])
           for name, j in fields.items()},
        sn_log_t_myr=log[:, 0] / MYR_CODE, sn_log_z_pc=log[:, 1] - Z0, sn_log_n_h=log[:, 2],
        sn_log_thermal=log[:, 3].astype(bool),
        mode=MODE, dx_pc=DX, z0_pc=Z0, n_cells=np.array([N_XY, N_XY, N_Z]), wall_time_s=wall,
        sn_injection_radius_pc=SN_INJECTION_RADIUS,
        mhd=MHD, self_gravity=SELF_GRAVITY, jeans_floor=JEANS_FLOOR,
        code_pressure_erg_cm3=CODE_PRESSURE_ERG_CM3,
        code_density_g_cm3=_code(1 * CODE_DENSITY, u.g / u.cm**3),
        code_energy_density_erg_cm3=_code(1 * CODE_UNITS.code_energy / CODE_UNITS.code_length**3, u.erg / u.cm**3),
        code_temperature_per_kelvin=KI_PARAMS.code_temperature_per_kelvin,
        state_index=np.array([rv.density_index, rv.velocity_index.x, rv.velocity_index.y, rv.velocity_index.z,
                              rv.pressure_index] + ([rv.cosmic_ray_e_index] if CR_ACTIVE else [])),
    )
    print(f"raw results written to {path} ({path.stat().st_size / 1e9:.2f} GB)")


def run_m7():
    config = _base_config()
    registered_variables = get_registered_variables(config)
    phi, initial_state, helper_data = _potential_and_ic(config, registered_variables)
    config = finalize_config(config, initial_state.shape)

    z_rel = np.asarray(helper_data.geometric_centers[0, 0, :, 2]) - Z0
    mid = int(np.argmin(np.abs(z_rel)))
    t_ic = np.asarray(get_temperature_from_pressure(
        initial_state[registered_variables.density_index], initial_state[registered_variables.pressure_index],
        X_H, Z_METAL,
    ))[0, 0, :] / KI_PARAMS.code_temperature_per_kelvin
    print(f"mode = {MODE}: E_thermal = {E_THERMAL_ERG:.1e} erg, E_CR = {E_CR_ERG:.1e} erg per SN; "
          f"CR transport {'off' if not CR_ACTIVE else ('anisotropic (along B)' if MHD else 'isotropic')}; "
          f"MHD {'on (B0 = 3 muG = %.3f code, tolerance %s)' % (np.sqrt(2 * P_B_MIDPLANE_CODE), MHD_TOLERANCE) if MHD else 'off'}; "
          f"self-gravity {'on (G = %.4g code), Jeans floor %s' % (G_CODE, 'on' if JEANS_FLOOR else 'OFF') if SELF_GRAVITY else 'off'}")
    print(f"grid {N_XY} x {N_XY} x {N_Z}, dx = {DX:.2f} pc; midplane n_H = {RHO_MIDPLANE * N_H_PER_CODE_DENSITY:.3f} cm^-3, "
          f"T_IC(midplane) = {t_ic[mid]:.3g} K, T_IC(z=+300 pc) = {t_ic[int(np.argmin(np.abs(z_rel - 300)))]:.3g} K, "
          f"T_IC(top) = {t_ic[-1]:.3g} K")
    print(f"SN rate = {SN_RATE_CODE * MYR_CODE:.1f} /Myr, injection radius {SN_INJECTION_RADIUS:.1f} pc "
          f"= {SN_INJECTION_RADIUS / DX:.2f} cells")
    if SN_INJECTION_RADIUS < 1.5 * DX:
        print(f"[warning] SN injection radius is only {SN_INJECTION_RADIUS / DX:.2f} cells at this --res; "
              f"deposits are barely resolved")
    if E51 > 0:
        n_resolved = (SN_SHELL_FORMATION_RADIUS_COEFFICIENT / SN_INJECTION_RADIUS) ** (1 / 0.42)
        print(f"shell-formation radius at n_H=1: {SN_SHELL_FORMATION_RADIUS_COEFFICIENT:.1f} pc -> thermal "
              f"injection below n_H ~ {n_resolved:.3g} cm^-3, momentum above")
    if CR_ACTIVE:
        print(f"kappa = 1e28 cm^2/s = {KAPPA_CODE:.4g} code, v_red = {V_RED_CODE:.0f} km/s, "
              f"diffusion length over t_end = {np.sqrt(KAPPA_CODE * T_END) / 1000.0:.2f} kpc")
    _cost_estimate()
    if SETUP_ONLY:
        return None

    params = SimulationParams(
        t_end=T_END,
        gamma=GAMMA,
        minimum_density=MIN_DENSITY_CODE,
        minimum_pressure=MIN_PRESSURE_CODE,
        cooling_params=CoolingParams(
            hydrogen_mass_fraction=X_H,
            metal_mass_fraction=Z_METAL,
            floor_temperature=FLOOR_TEMPERATURE_CODE,
            jeans_floor_cells=JEANS_FLOOR_CELLS,
            cooling_curve_params=KI_PARAMS,
        ),
        sn_driving_params=SNDrivingParams(
            sn_rate=SN_RATE_CODE,
            sn_energy=SN_ENERGY_CODE,
            sn_cr_fraction=SN_CR_FRACTION,
            sn_injection_radius=SN_INJECTION_RADIUS,
            sn_smooth_cells=SN_SMOOTH_CELLS,
            sn_momentum_coefficient=SN_MOMENTUM_COEFFICIENT,
            sn_momentum_density_reference=SN_MOMENTUM_DENSITY_REFERENCE,
            sn_momentum_thermal_floor_fraction=SN_MOMENTUM_THERMAL_FLOOR,
            sn_shell_formation_radius_coefficient=SN_SHELL_FORMATION_RADIUS_COEFFICIENT,
            sn_gaussian_z_center=Z0,
            sn_gaussian_scale_height_1=SN_TYPE_II_HEIGHT,
            sn_gaussian_scale_height_2=SN_TYPE_IA_HEIGHT,
            sn_gaussian_first_fraction=SN_TYPE_II_FRACTION,
        ),
        cosmic_ray_grey_params=CosmicRayGreyParams(
            gamma_cr=GAMMA_CR,
            diffusion_coefficient=KAPPA_CODE,
            reduced_streaming_speed=V_RED_CODE,
        ),
    )
    params = params._replace(gravitational_potential=phi, gravitational_constant=G_CODE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)  # fail now, not after the run
    SN_EVENT_LOG.clear()
    print("Starting M7 pilot run...")
    wall_start = time.time()
    snapshot_data = time_integration(initial_state, config, params, registered_variables)
    wall = time.time() - wall_start
    print(f"wall time {wall / 3600.0:.2f} h for {T_END / MYR_CODE:.0f} Myr "
          f"({wall / (T_END / MYR_CODE):.1f} s per simulated Myr, compilation included)")
    _save_run_data(snapshot_data, registered_variables, wall)

    t_myr = np.asarray(snapshot_data.time_points) / MYR_CODE
    states = snapshot_data.states
    rows = []
    print("\n--- per-snapshot summary ---")
    for i, t in enumerate(t_myr):
        state = states[i]
        rho = np.asarray(state[registered_variables.density_index])
        if np.any(np.isnan(np.asarray(state))) or np.all(rho == 0):
            print(f"  snapshot {i:3d}  t={t:7.1f} Myr   NaN/corrupted state")
            rows.append([np.nan] * 11)
            continue
        temp = np.asarray(get_temperature_from_pressure(
            state[registered_variables.density_index], state[registered_variables.pressure_index], X_H, Z_METAL,
        )) / KI_PARAMS.code_temperature_per_kelvin
        n_mid = float(rho[:, :, mid].mean() * N_H_PER_CODE_DENSITY)
        z70, z90 = _mass_heights(rho, z_rel)
        mdot, v_out, eta = _outflow(state, registered_variables, z_rel)
        p_gas_z = np.asarray(state[registered_variables.pressure_index]).mean(axis=(0, 1))
        h_gas = _scale_height(p_gas_z, z_rel, mid)
        if CR_ACTIVE:
            p_cr_z = (GAMMA_CR - 1.0) * np.asarray(state[registered_variables.cosmic_ray_e_index]).mean(axis=(0, 1))
            h_cr = _scale_height(p_cr_z, z_rel, mid)
        else:
            h_cr = np.nan
        if MHD:
            b = np.asarray(state[registered_variables.magnetic_index.x:registered_variables.magnetic_index.z + 1])
            b_sq = np.sum(b**2, axis=0)
            # B^2 / 2 (code) = B_G^2 / (8 pi) (erg/cm^3)  ->  B_G = sqrt(4 pi B^2 P_unit)
            b_mid_mugauss = float(np.sqrt(4 * np.pi * b_sq[:, :, mid].mean() * CODE_PRESSURE_ERG_CM3)) * 1e6
            v_alfven_max = float(np.sqrt(b_sq / rho).max())
        else:
            b_mid_mugauss = v_alfven_max = np.nan
        # columns: n_mid, max T, z70, z90, mdot, v_out, eta, H_gas, H_cr, B_rms(mid) [muG], max v_A [km/s]
        rows.append([n_mid, float(temp.max()), z70, z90, mdot, v_out, eta, h_gas, h_cr, b_mid_mugauss, v_alfven_max])
        print(f"  snapshot {i:3d}  t={t:7.1f} Myr   n_H(mid)={n_mid:8.3f}  max(T)={temp.max():9.3e} K  "
              f"z70={z70:7.1f} pc  z90={z90:7.1f} pc  eta(1 kpc)={eta: .3e}  v_out={v_out:7.2f} km/s  "
              f"H_gas={h_gas:7.1f}  H_cr={h_cr:7.1f} pc"
              + (f"  B_rms(mid)={b_mid_mugauss:6.2f} muG  max v_A={v_alfven_max:8.1f} km/s" if MHD else ""))
    rows = np.array(rows, dtype=float)

    pics = Path(__file__).resolve().parent / "pics"
    pics.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, j, ylabel, log in [
        (axes[0, 0], 0, r"midplane $\langle n_H\rangle$ (cm$^{-3}$)", True),
        (axes[0, 1], 1, "max T (K)", True),
        (axes[0, 2], 2, "height enclosing 70% / 90% of the mass (pc)", True),
        (axes[1, 0], 6, r"mass loading $\eta$ at $|z|$ = 1 kpc", True),
        (axes[1, 1], 5, r"$v_{\rm out}$ at $|z|$ = 1 kpc (km/s)", False),
        (axes[1, 2], 7, r"pressure scale height (pc)", False),
    ]:
        plot = ax.semilogy if log else ax.plot
        plot(t_myr, rows[:, j], "-o", ms=3, label="z70" if j == 2 else ("gas" if j == 7 else None))
        if j == 2:
            plot(t_myr, rows[:, 3], "-s", ms=3, label="z90")
        if j == 7 and CR_ACTIVE:
            plot(t_myr, rows[:, 8], "-s", ms=3, label="CR")
        if j in (2, 7):
            ax.legend()
        ax.set(xlabel="t (Myr)", ylabel=ylabel)
    fig.suptitle(f"M7 pilot ({MODE}, {PHYSICS_LABEL}), dx = {DX:.1f} pc")
    fig.tight_layout()
    fig.savefig(pics / f"m7_pilot_timeseries{OUT_SUFFIX}.svg")
    plt.close(fig)

    _sn_mode_report(t_myr, pics)

    valid = np.where(~np.isnan(rows[:, 0]))[0]
    if valid.size:
        last = int(valid[-1])
        _structure_plot(states[last], registered_variables, mid, t_myr[last],
                        pics / f"m7_pilot_structure_girichidis_fig1{OUT_SUFFIX}.svg")
        _edge_on_animation(states, registered_variables, mid, t_myr, [int(i) for i in valid],
                           pics / f"m7_pilot_edge_on_density{OUT_SUFFIX}.gif")
        tail = valid[-max(1, valid.size // 4):]
        print("\n--- late-time averages (last quarter of valid snapshots) ---")
        print(f"  z70 / z90 = {np.nanmean(rows[tail, 2]):.0f} / {np.nanmean(rows[tail, 3]):.0f} pc "
              f"(Girichidis 2016, CR runs: ~200 / ~1500 pc)")
        print(f"  eta(1 kpc) = {np.nanmean(rows[tail, 6]):.3g}, v_out = {np.nanmean(rows[tail, 5]):.1f} km/s")
        def tail_mean(j):
            values = rows[tail, j]
            return f"{np.nanmean(values):.0f} pc" if np.any(~np.isnan(values)) else "undefined"
        # H_cr is NaN once the CR pressure falls by < 1/e within the box
        print(f"  H_gas = {tail_mean(7)}, H_cr = {tail_mean(8)}")
        print(f"plots written to {pics}/m7_pilot_*{OUT_SUFFIX}.svg/.gif")
    return snapshot_data


if __name__ == "__main__":
    run_m7()
