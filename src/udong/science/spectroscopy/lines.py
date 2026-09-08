"""Constrained multi-Gaussian emission-line fitting.

Implements the emission-line *model* used by Cao+ (2026, arXiv:2606.24211,
Section II.2.2) for their MaNGA analysis.  After stellar-continuum
subtraction the strongest optical lines are fitted with one or two Gaussian
components with **explicit** constraints:

* lines fitted in the same window share one velocity centroid per component;
* the dispersion of each non-reference line is a free factor of the reference
  (Halpha) dispersion, bounded to 0.75-1.25 (paper default);
* doublet integrated-flux ratios are fixed to the theoretical values
  ([N II] 6548/6585 = 1/2.96, [O III] 4959/5007 = 1/2.98);
* the 1- vs 2-component choice is made with
  ``Delta BIC = BIC(1 comp) - BIC(2 comp) > 10`` (paper threshold).

Fitting is *window based*: a :class:`LineWindow` lists the lines fitted
simultaneously over a short rest-frame interval of the continuum-subtracted
spectrum.

Explicit assumptions
--------------------
* The input is **rest-frame, continuum-subtracted** flux density (per
  Angstrom); a tiny additive Legendre baseline (default linear) absorbs any
  residual continuum mismatch.
* Profiles are Gaussians in flux density.  A component has one velocity
  ``v`` (km/s) and one reference dispersion ``sigma`` (km/s, observed /
  uncorrected for the instrumental LSF); the wavelength-space dispersion of
  line ``l`` is ``sigma_AA = sigma * factor_l * lambda_rest(l) / c``.
* Integrated line flux is the fitted quantity; the Gaussian peak is
  ``F / (sigma_AA * sqrt(2 pi))`` so fixed integrated doublet ratios are exact.
* Errors are Gaussian (``1/sqrt(ivar)``); parameter uncertainties are obtained
  from the least-squares Jacobian (linearised, adequate for S/N cuts).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, lsq_linear

__all__ = [
    "C_KM",
    "Line",
    "LineWindow",
    "GaussianComponent",
    "WindowFit",
    "fit_window",
    "fit_single_and_double",
    "evaluate_components",
    "fit_fixed_kinematics",
    "bic",
    "HA_NII_WINDOW",
    "OIII_WINDOW",
    "HB_WINDOW",
    "SII_WINDOW",
]

C_KM = 299792.458  # speed of light, km/s


@dataclass(frozen=True)
class Line:
    """One emission line of a :class:`LineWindow`.

    ``ratio_ref``
        Name of another line of the window whose **integrated flux** this line
        is tied to.  ``None`` means the amplitude is free.
    ``ratio``
        Integrated-flux ratio relative to ``ratio_ref`` (used only when
        ``ratio_ref`` is set).
    ``sigma_factor``
        ``"same"``: dispersion identical to the reference line; ``"free"``:
        dispersion is a free factor bounded by ``LineWindow.sigma_factor_bounds``.
    """

    name: str
    wavelength: float  # rest-frame wavelength, AA
    ratio_ref: str | None = None
    ratio: float = 1.0
    sigma_factor: str = "free"


@dataclass(frozen=True)
class LineWindow:
    """A short rest-frame wavelength window and its line list."""

    name: str
    wavelength_lo: float  # AA, rest frame
    wavelength_hi: float
    lines: tuple[Line, ...]
    sigma_factor_bounds: tuple[float, float] = (0.75, 1.25)

    def __post_init__(self) -> None:
        if not self.lines:
            raise ValueError("window needs at least one line")
        names = {ln.name for ln in self.lines}
        free = [ln for ln in self.lines if ln.ratio_ref is None]
        if len(free) == 0:
            raise ValueError("at least one line must have a free amplitude")
        for ln in self.lines:
            if ln.ratio_ref is not None:
                if ln.ratio_ref not in names:
                    raise ValueError(f"line {ln.name!r}: unknown ratio_ref {ln.ratio_ref!r}")
                ref = next(x for x in self.lines if x.name == ln.ratio_ref)
                if ref.ratio_ref is not None:
                    raise ValueError("ratio_ref must point to a free-amplitude line")
            if ln.sigma_factor not in ("same", "free"):
                raise ValueError(f"unknown sigma_factor {ln.sigma_factor!r}")
        if self.lines[0].sigma_factor != "same":
            raise ValueError("first (reference) line must use sigma_factor='same'")


# --- physical windows used for the 4C+29.30 reproduction ------------------- #
# vacuum rest wavelengths as used by the paper (MaNGA WAVE is vacuum).
HA_NII_WINDOW = LineWindow(
    "Ha+NII",
    6525.0, 6625.0,
    (
        Line("Ha", 6564.61, sigma_factor="same"),
        Line("NII6548", 6549.86, ratio_ref="NII6585", ratio=1.0 / 2.96),
        Line("NII6585", 6585.27),
    ),
)
OIII_WINDOW = LineWindow(
    "OIII",
    4945.0, 5040.0,
    (
        Line("OIII5007", 5008.24, sigma_factor="same"),
        Line("OIII4959", 4960.30, ratio_ref="OIII5007", ratio=1.0 / 2.98,
             sigma_factor="same"),
    ),
)
HB_WINDOW = LineWindow(
    "Hb", 4825.0, 4900.0,
    (Line("Hb", 4862.68, sigma_factor="same"),),
)
SII_WINDOW = LineWindow(
    "SII", 6690.0, 6760.0,
    (
        Line("SII6718", 6718.29, sigma_factor="same"),
        Line("SII6732", 6732.67, sigma_factor="same"),
    ),
)
WINDOWS: dict[str, LineWindow] = {
    "Ha+NII": HA_NII_WINDOW,
    "OIII": OIII_WINDOW,
    "Hb": HB_WINDOW,
    "SII": SII_WINDOW,
}


@dataclass
class GaussianComponent:
    """One fitted kinematic component (sorted by increasing sigma)."""

    velocity: float  # km/s
    sigma: float  # km/s, observed
    fluxes: dict[str, float]  # integrated flux per line (window units)
    flux_sigma: dict[str, float] | None = None


@dataclass
class WindowFit:
    """Result of fitting one window."""

    window: LineWindow
    n_components: int
    components: list[GaussianComponent]
    chi2: float
    dof: int
    bic: float
    success: bool
    message: str
    n_eval: int = 0
    param_names: list[str] | None = None

    @property
    def reduced_chi2(self) -> float:
        return self.chi2 / max(self.dof, 1)


def bic(chi2: float, n_params: int, n_pixels: int) -> float:
    """Gaussian-noise BIC: ``chi2 + k ln(n)``.

    The same definition is used for the 1- and 2-component models so additive
    constants cancel in ``Delta BIC``.
    """
    return chi2 + n_params * np.log(max(int(n_pixels), 2))


def _line_sigma_aa(sigma_km_s: float, factor: float, wavelength_aa: float) -> float:
    return sigma_km_s * factor * wavelength_aa / C_KM


def _gaussian_flux_density(lam: np.ndarray, lam_c: float, sig_aa: float, flux: float):
    peak = flux / (sig_aa * np.sqrt(2.0 * np.pi))
    return peak * np.exp(-0.5 * ((lam - lam_c) / sig_aa) ** 2)


class _Problem:
    """Parameter bookkeeping for a window fit.

    Layout of the plain (unconstrained) parameter vector:

    * ``v``            n_comp velocities (km/s)
    * ``sigma``        n_comp dispersions (km/s)
    * ``sigma_factors`` free factors (one per component & "free" line)
    * ``log_flux``     log of the integrated flux of the *free-amplitude*
                       lines (one per component)
    * ``baseline``     (1 + baseline_degree) additive coefficients
    """

    def __init__(self, window: LineWindow, n_components: int, baseline_degree: int):
        self.window = window
        self.n_comp = n_components
        self.n_lines = len(window.lines)
        self.n_base = 1 + baseline_degree
        # per-component free amplitude lines
        self.free_amp = [
            [k for k, ln in enumerate(window.lines) if ln.ratio_ref is None]
            for _ in range(n_components)
        ]
        # tie map: line index -> (free-anchor line index, log(ratio))
        self.ties: dict[int, tuple[int, float]] = {}
        for k, ln in enumerate(window.lines):
            if ln.ratio_ref is not None:
                anchor = next(i for i, x in enumerate(window.lines) if x.name == ln.ratio_ref)
                self.ties[k] = (anchor, np.log(ln.ratio))
        # free sigma-factor entries: (component, line)
        self.free_sigma = [
            (j, k)
            for j in range(n_components)
            for k, ln in enumerate(window.lines)
            if ln.sigma_factor == "free"
        ]
        self.n_free_sigma = len(self.free_sigma)
        self.n_free_amp = len(self.free_amp[0])

    @property
    def n_params(self) -> int:
        return (
            2 * self.n_comp
            + self.n_free_sigma
            + self.n_comp * self.n_free_amp
            + self.n_base
        )

    # -- helpers ------------------------------------------------------- #
    def _free_flux_matrix(self, log_flux):
        """Full (n_comp, n_lines) log-flux matrix from the free amplitudes."""
        lf = np.zeros((self.n_comp, self.n_lines))
        for j in range(self.n_comp):
            for i, k in enumerate(self.free_amp[j]):
                lf[j, k] = log_flux[j, i]
        for k, (anchor, lr) in self.ties.items():
            for j in range(self.n_comp):
                lf[j, k] = lf[j, anchor] + lr
        return lf

    def pack(self, v, sigma, factors, log_flux, baseline) -> np.ndarray:
        fac = np.array([factors[j, k] for j, k in self.free_sigma])
        return np.concatenate(
            [np.asarray(v), np.asarray(sigma), fac,
             np.asarray(log_flux).ravel(), np.asarray(baseline)]
        )

    def unpack(self, p: np.ndarray):
        i = 0
        v = p[i:i + self.n_comp]
        i += self.n_comp
        sigma = p[i:i + self.n_comp]
        i += self.n_comp
        factors = np.ones((self.n_comp, self.n_lines))
        for (j, k), val in zip(self.free_sigma, p[i:i + self.n_free_sigma], strict=True):
            factors[j, k] = val
        i += self.n_free_sigma
        log_flux = p[i:i + self.n_comp * self.n_free_amp].reshape(
            self.n_comp, self.n_free_amp)
        i += self.n_comp * self.n_free_amp
        baseline = p[i:i + self.n_base]
        return v, sigma, factors, log_flux, baseline

    def names(self) -> list[str]:
        names = []
        for j in range(self.n_comp):
            names += [f"v{j}", f"sigma{j}"]
        for j, k in self.free_sigma:
            names.append(f"factor_c{j}_{self.window.lines[k].name}")
        for j in range(self.n_comp):
            for k in self.free_amp[j]:
                names.append(f"lnF_c{j}_{self.window.lines[k].name}")
        for i in range(self.n_base):
            names.append(f"base{i}")
        return names

    def model(self, lam, v, sigma, factors, log_flux, baseline):
        lf = self._free_flux_matrix(log_flux)
        out = np.zeros_like(lam, dtype=float)
        if self.n_base:
            lam0 = 0.5 * (self.window.wavelength_lo + self.window.wavelength_hi)
            scale = max(0.5 * (self.window.wavelength_hi - self.window.wavelength_lo), 1e-6)
            out += np.polynomial.legendre.legval((lam - lam0) / scale, baseline)
        for j in range(self.n_comp):
            for k, line in enumerate(self.window.lines):
                flux = np.exp(lf[j, k])
                lam_c = line.wavelength * (1.0 + v[j] / C_KM)
                sig_aa = _line_sigma_aa(sigma[j], factors[j, k], line.wavelength)
                out += _gaussian_flux_density(lam, lam_c, sig_aa, flux)
        return out


def _finite_jacobian(fun, p0, eps=1e-6):
    """Central-difference Jacobian of ``fun`` at ``p0``."""
    p0 = np.asarray(p0, dtype=float)
    f0 = np.asarray(fun(p0), dtype=float)
    jac = np.empty((f0.size, p0.size))
    for i in range(p0.size):
        step = eps * max(abs(p0[i]), 1.0)
        pp = p0.copy()
        pm = p0.copy()
        pp[i] += step
        pm[i] -= step
        jac[:, i] = (np.asarray(fun(pp)) - np.asarray(fun(pm))) / (2 * step)
    return jac


def _default_seed(window, lam, f, n_comp, init, sigma_bounds):
    n_lines = len(window.lines)
    if init is not None:
        v0 = np.broadcast_to(np.atleast_1d(init.get("velocity", 0.0)), (n_comp,)).astype(float)
        s0 = np.broadcast_to(np.atleast_1d(init.get("sigma", 120.0)), (n_comp,)).astype(float)
        flux0 = np.asarray(init.get("flux", None))
    else:
        v0 = np.zeros(n_comp)
        s0 = np.array([120.0, 300.0]) if n_comp == 2 else np.array([120.0])
        flux0 = None
    if flux0 is None or flux0.shape != (n_comp, n_lines):
        flux0 = np.zeros((n_comp, n_lines))
        for k, ln in enumerate(window.lines):
            if ln.ratio_ref is not None:
                continue
            m = np.abs(lam - ln.wavelength) < 300.0 * ln.wavelength / C_KM
            if m.any():
                flux0[:, k] = max(float(np.max(f[m])), 1e-9)
    # keep only free-amplitude entries
    free_idx = [k for k, ln in enumerate(window.lines) if ln.ratio_ref is None]
    lf0 = np.log(np.clip(flux0[:, free_idx], 1e-9, None))
    if n_comp == 2 and np.allclose(v0, v0[0]):
        v0 = np.array([v0[0] - 30.0, v0[0] + 30.0])
    return v0, s0, lf0


def fit_window(
    wavelength: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    window: LineWindow,
    *,
    n_components: int = 1,
    v_bounds: tuple[float, float] = (-1000.0, 1000.0),
    sigma_bounds: tuple[float, float] = (25.0, 700.0),
    baseline_degree: int = 1,
    init: dict | None = None,
) -> WindowFit:
    """Fit ``window`` on a rest-frame continuum-subtracted spectrum.

    ``wavelength/flux/ivar`` are 1-D arrays covering the window (pixels with
    ``ivar <= 0`` or outside the window are ignored).  ``init`` may provide
    ``velocity`` / ``sigma`` / ``flux`` seeds.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)
    ivar = np.asarray(ivar, dtype=float)
    if wavelength.ndim != 1 or flux.shape != wavelength.shape or ivar.shape != wavelength.shape:
        raise ValueError("wavelength/flux/ivar must be 1-D arrays of the same size")

    sel = (
        np.isfinite(wavelength) & np.isfinite(flux) & np.isfinite(ivar)
        & (ivar > 0) & (wavelength >= window.wavelength_lo)
        & (wavelength <= window.wavelength_hi)
    )
    lam = wavelength[sel]
    f = flux[sel]
    noise = 1.0 / np.sqrt(ivar[sel])
    n_pix = lam.size
    if n_pix < 20:
        return WindowFit(window, n_components, [], np.nan, 0, np.nan, False,
                         f"too few good pixels ({n_pix})")

    prob = _Problem(window, n_components, baseline_degree)
    n_comp = n_components

    v0, s0, lf0 = _default_seed(window, lam, f, n_comp, init, sigma_bounds)
    base0 = np.zeros(prob.n_base)
    fac_full = np.ones((n_comp, len(window.lines)))  # factors matrix (pack extracts free)

    p0 = prob.pack(v0, s0, fac_full, lf0, base0)

    # ---------- bounds ----------
    lo = [v_bounds[0]] * n_comp + [sigma_bounds[0]] * n_comp
    lo += [window.sigma_factor_bounds[0]] * prob.n_free_sigma
    lo += [-25.0] * (n_comp * prob.n_free_amp)
    lo += [-np.inf] * prob.n_base
    hi = [v_bounds[1]] * n_comp + [sigma_bounds[1]] * n_comp
    hi += [window.sigma_factor_bounds[1]] * prob.n_free_sigma
    hi += [np.log(1e7)] * (n_comp * prob.n_free_amp)
    hi += [np.inf] * prob.n_base

    def resid_from(p):
        v, sigma, factors, lf, base = prob.unpack(p)
        return (prob.model(lam, v, sigma, factors, lf, base) - f) / noise

    if n_comp == 2:
        # enforce sigma2 >= sigma1 by fitting sigma1 + exp(delta)
        def resid_ordered(p):
            v = p[:2]
            s1 = p[2]
            sigma = np.array([s1, s1 + np.exp(p[3])])
            full = np.concatenate([v, sigma, p[4:]])
            v2, s2, factors, lf, base = prob.unpack(full)
            return (prob.model(lam, v2, s2, factors, lf, base) - f) / noise

        s_min, s_max = float(min(s0)), float(max(s0))
        delta0 = float(np.log(max(s_max - s_min, 20.0)))
        p0 = np.concatenate([v0, [s_min, delta0], p0[2 * n_comp:]])
        lo = lo[:2] + [sigma_bounds[0], np.log(10.0)] + lo[2 * n_comp:]
        hi = hi[:2] + [sigma_bounds[1], np.log(2000.0)] + hi[2 * n_comp:]
        resfun = resid_ordered
        unorder = lambda p: np.concatenate([p[:2], [p[2], p[2] + np.exp(p[3])], p[4:]])  # noqa: E731
    else:
        resfun = resid_from
        unorder = lambda p: p  # noqa: E731

    p0 = np.clip(p0, np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))
    try:
        res = least_squares(resfun, p0, bounds=(np.asarray(lo), np.asarray(hi)),
                            xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=2000)
        success = bool(res.success)
        message = str(res.message)
    except Exception as exc:  # noqa: BLE001
        return WindowFit(window, n_components, [], np.nan, 0, np.nan, False, str(exc))

    p_plain = np.asarray(unorder(res.x), dtype=float)
    vf, sf, factors, lf, _ = prob.unpack(p_plain)
    lf_full = prob._free_flux_matrix(lf)

    # linearised flux uncertainties from the Jacobian of the weighted residual
    flux_sigma = None
    try:
        jac = _finite_jacobian(resid_from, p_plain)
        cov = np.linalg.inv(jac.T @ jac)
        dof_cov = max(res.fun.size - p_plain.size, 1)
        cov = cov * max(float(np.sum(res.fun ** 2)) / dof_cov, 1.0)
    except Exception:  # noqa: BLE001
        cov = None
    if cov is not None:
        flux_sigma = {}
        for j in range(n_comp):
            errs = {}
            for i, k in enumerate(prob.free_amp[j]):
                idx = 2 * n_comp + prob.n_free_sigma + j * prob.n_free_amp + i
                if idx < cov.shape[0]:
                    var = max(float(cov[idx, idx]), 0.0)
                    errs[window.lines[k].name] = float(np.exp(lf[j, i]) * np.sqrt(var))
            # tied lines share the anchor fractional error
            for k, (anchor, _lr) in prob.ties.items():
                if window.lines[anchor].name in errs:
                    anchor_name = window.lines[anchor].name
                    errs[window.lines[k].name] = errs[anchor_name] * float(
                        np.exp(lf_full[j, k]) / np.exp(lf_full[j, anchor]))
            flux_sigma[f"c{j}"] = errs

    order = np.argsort(sf)
    components = []
    for j in order:
        fluxes = {ln.name: float(np.exp(lf_full[j, k])) for k, ln in enumerate(window.lines)}
        errs = flux_sigma[f"c{j}"] if flux_sigma is not None else None
        components.append(GaussianComponent(float(vf[j]), float(sf[j]), fluxes, errs))

    chi2 = float(np.sum(res.fun ** 2))
    n_params = len(p_plain)
    dof = n_pix - n_params
    return WindowFit(window, n_components, components, chi2, dof,
                     bic(chi2, n_params, n_pix), success, message,
                     n_eval=int(getattr(res, "nfev", 0)),
                     param_names=prob.names())


def fit_single_and_double(
    wavelength: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    window: LineWindow,
    **kwargs,
) -> tuple[WindowFit, WindowFit]:
    """Run the 1- and 2-component fits of ``window``."""
    single = fit_window(wavelength, flux, ivar, window, n_components=1, **kwargs)
    double = fit_window(wavelength, flux, ivar, window, n_components=2, **kwargs)
    return single, double


def fit_fixed_kinematics(
    wavelength: np.ndarray,
    flux: np.ndarray,
    ivar: np.ndarray,
    window: LineWindow,
    components: list[tuple[float, float]],
    *,
    baseline_degree: int = 1,
) -> WindowFit:
    """Linear (amplitude-only) fit with **fixed** component kinematics.

    Used to measure the flux of lines (e.g. Hbeta, [S II]) whose kinematics
    are tied to a reference system (e.g. the Halpha components) without
    letting the weak line drive its own kinematics.  ``components`` is a list
    of ``(velocity_km_s, sigma_km_s)``; line dispersions are those of the
    component and the fixed doublet ratios of the window are enforced.

    The fit is a bounded linear least squares (amplitudes >= 0, baseline free).
    """
    wavelength = np.asarray(wavelength, dtype=float)
    flux = np.asarray(flux, dtype=float)
    ivar = np.asarray(ivar, dtype=float)
    sel = (
        np.isfinite(wavelength) & np.isfinite(flux) & np.isfinite(ivar)
        & (ivar > 0) & (wavelength >= window.wavelength_lo)
        & (wavelength <= window.wavelength_hi)
    )
    lam = wavelength[sel]
    f = flux[sel]
    noise = 1.0 / np.sqrt(ivar[sel])
    n_pix = lam.size
    if n_pix < 20:
        return WindowFit(window, len(components), [], np.nan, 0, np.nan, False,
                         "too few good pixels")

    n_base = 1 + baseline_degree
    lam0 = 0.5 * (window.wavelength_lo + window.wavelength_hi)
    scale = max(0.5 * (window.wavelength_hi - window.wavelength_lo), 1e-6)
    x = (lam - lam0) / scale

    # free-amplitude lines per component (tied lines are fixed ratios)
    free_idx = [k for k, ln in enumerate(window.lines) if ln.ratio_ref is None]
    n_free = len(free_idx)
    n_comp = len(components)

    def make_model(amps):
        out = np.zeros_like(lam)
        for j, (vj, sj) in enumerate(components):
            for k, line in enumerate(window.lines):
                flux_here = amps[j * n_free + free_idx.index(k)] if k in free_idx else None
                if flux_here is None:
                    continue  # handled through the anchor scaling below
                lam_c = line.wavelength * (1.0 + vj / C_KM)
                sig_aa = _line_sigma_aa(sj, 1.0, line.wavelength)
                out += _gaussian_flux_density(lam, lam_c, sig_aa, flux_here)
        # tied lines
        for j, (vj, sj) in enumerate(components):
            for line in window.lines:
                if line.ratio_ref is None:
                    continue
                anchor = next(i for i, ln in enumerate(window.lines) if ln.name == line.ratio_ref)
                f_anchor = amps[j * n_free + free_idx.index(anchor)]
                lam_c = line.wavelength * (1.0 + vj / C_KM)
                sig_aa = _line_sigma_aa(sj, 1.0, line.wavelength)
                out += _gaussian_flux_density(lam, lam_c, sig_aa, f_anchor * line.ratio)
        # baseline (linear in normalized coordinate)
        if n_base == 2:
            out += amps[n_comp * n_free] + amps[n_comp * n_free + 1] * x
        else:
            out += amps[n_comp * n_free]
        return out

    n_col = n_comp * n_free + n_base
    design = np.empty((n_pix, n_col))
    for i in range(n_col):
        e = np.zeros(n_col)
        e[i] = 1.0
        design[:, i] = make_model(e)
    design /= noise[:, None]
    y = f / noise
    bounds = (np.concatenate([np.zeros(n_comp * n_free), [-np.inf] * n_base]),
              np.concatenate([np.full(n_comp * n_free, 1e7), [np.inf] * n_base]))
    try:
        sol = lsq_linear(design, y, bounds=bounds, method="trf")
        success = sol.success
        message = str(sol.message) if hasattr(sol, "message") else ""
    except Exception as exc:  # noqa: BLE001
        return WindowFit(window, n_comp, [], np.nan, 0, np.nan, False, str(exc))

    amps = sol.x
    model = make_model(amps)
    chi2 = float(np.sum(((model - f) / noise) ** 2))
    # per-component flux dict (free + tied)
    comps = []
    for j in range(n_comp):
        fluxes = {}
        for k, ln in enumerate(window.lines):
            if ln.ratio_ref is None:
                fluxes[ln.name] = float(amps[j * n_free + free_idx.index(k)])
            else:
                anchor = next(i for i, x in enumerate(window.lines) if x.name == ln.ratio_ref)
                fluxes[ln.name] = fluxes[window.lines[anchor].name] * ln.ratio
        vj, sj = components[j]
        comps.append(GaussianComponent(vj, sj, fluxes, None))
    # parameter errors from the linear design
    try:
        cov = np.linalg.inv(design.T @ design)
    except Exception:  # noqa: BLE001
        cov = None
    if cov is not None:
        err = np.sqrt(np.maximum(np.diag(cov), 0.0))
        for j in range(n_comp):
            errs = {}
            for i, k in enumerate(free_idx):
                errs[window.lines[k].name] = float(err[j * n_free + i])
            for ln in window.lines:
                if ln.ratio_ref is not None:
                    anchor = next(i for i, x in enumerate(window.lines) if x.name == ln.ratio_ref)
                    errs[ln.name] = errs[window.lines[anchor].name] * ln.ratio
            comps[j].flux_sigma = errs

    n_params = n_col
    dof = n_pix - n_params
    return WindowFit(window, n_comp, comps, chi2, dof, bic(chi2, n_params, n_pix),
                     success, message)


def evaluate_components(
    wavelength: np.ndarray,
    window: LineWindow,
    components: list[GaussianComponent],
    baseline: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate a fitted window model (sum over components) at ``wavelength``.

    Useful for plotting; uses the stored component fluxes/velocities/sigmas
    and enforces the same doublet ties as the fit.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    out = np.zeros_like(wavelength, dtype=float)
    if baseline is not None and baseline.size:
        lam0 = 0.5 * (window.wavelength_lo + window.wavelength_hi)
        scale = max(0.5 * (window.wavelength_hi - window.wavelength_lo), 1e-6)
        out += np.polynomial.legendre.legval((wavelength - lam0) / scale, baseline)
    for comp in components:
        for line in window.lines:
            flux = comp.fluxes.get(line.name, 0.0)
            if flux <= 0:
                continue
            lam_c = line.wavelength * (1.0 + comp.velocity / C_KM)
            sig_aa = _line_sigma_aa(comp.sigma, 1.0, line.wavelength)
            out += _gaussian_flux_density(wavelength, lam_c, sig_aa, flux)
    return out



