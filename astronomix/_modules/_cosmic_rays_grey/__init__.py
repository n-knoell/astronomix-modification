"""Grey (energy-integrated) two-moment cosmic-ray physics module.

Evolves a cosmic-ray energy density ``e_cr`` and flux ``F_cr`` as independent
state variables (Jiang & Oh 2018; Thomas & Pfrommer 2019). This is the only
cosmic-ray model in astronomix -- an earlier single-scalar polytropic model
(``astronomix._modules._cosmic_rays``) was retired in favor of this one; see
DESIGN.md's "Consolidating with the old _cosmic_rays model" note.
See ``DESIGN.md`` in this directory for the state variables, equations,
operator-split ordering and open questions.
"""
