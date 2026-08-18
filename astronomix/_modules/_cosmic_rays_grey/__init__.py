"""Grey (energy-integrated) two-moment cosmic-ray physics module.

Evolves a cosmic-ray energy density ``e_cr`` and flux ``F_cr`` as independent
state variables (Jiang & Oh 2018; Thomas & Pfrommer 2019), separate from the
older single-scalar polytropic model in ``astronomix._modules._cosmic_rays``.
See ``DESIGN.md`` in this directory for the state variables, equations,
operator-split ordering and open questions.
"""
