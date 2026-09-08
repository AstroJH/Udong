"""Kinematics: analysis of velocity / dispersion *fields*.

This package operates on already-produced ``Map2D`` velocity and
velocity-dispersion maps (from MaNGA DAP, or estimated with
``udong.science.spectroscopy.moments``):

* radial profiles (rotation curve / dispersion profile),
* global (area/IVAR/flux-weighted) values,
* instrumental-LSF deconvolution,
* linear velocity-plane fits (v0 + sky-frame gradient PA).

Boundary: per-spectrum *estimation* of M1/M2 from a cube belongs to
``udong.science.spectroscopy.moments``; this package is about *analyzing* the
resulting 2-D fields.
"""

from __future__ import annotations

from udong.science.kinematics.measure import (
    VelocityPlaneFit,
    fit_velocity_plane,
    global_dispersion,
    global_mean_velocity,
    lsf_correct_sigma,
    radial_dispersion_profile,
    radial_velocity_profile,
)

__all__ = [
    "VelocityPlaneFit",
    "fit_velocity_plane",
    "global_dispersion",
    "global_mean_velocity",
    "lsf_correct_sigma",
    "radial_dispersion_profile",
    "radial_velocity_profile",
]
