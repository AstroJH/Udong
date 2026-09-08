"""Plotting (thin matplotlib wrappers).

Plotting logic lives here, not scattered through science modules.
"""

from __future__ import annotations

from udong.viz.diagnostics import plot_bpt
from udong.viz.plots import plot_map, plot_profile, plot_spectrum

__all__ = ["plot_bpt", "plot_map", "plot_profile", "plot_spectrum"]
