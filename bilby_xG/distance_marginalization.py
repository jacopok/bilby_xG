"""Distance marginalisation by quadrature, accurate at any SNR.

At fixed other parameters the log-likelihood ratio is a quadratic in
u = d_ref / d_L,

    ln L(u) = a u - b u^2 / 2,    a = Re<d|h_ref>,  b = <h_ref|h_ref>,

with h_ref the template at d_ref: a Gaussian in u, of width 1 / sqrt(b),
peaked at u* = a / b. The marginal likelihood is

    ln Z_d = ln int p(d) exp(a u - b u^2 / 2) dd

over the distance prior's support. bilby interpolates it from a table: the
distance integral on 10^4 points uniform in d, the result splined on a
400 x 800 grid over 15 decades of a and b. At ET signal-to-noise ratios the
Gaussian in d is narrower than that grid (d / SNR = 0.08 Mpc at 40 Mpc and
SNR 500, against a step of 0.4 Mpc on 25-4000 Mpc), and the spline is coarse in
a and b. Here the integral is computed directly, by Gauss-Legendre in ln d:
the window where the Gaussian factor is within e^{-K^2/2} of its maximum
over the prior, cut into N_CHUNKS chunks of equal width in u (at most
2 / sqrt(b) each, over which the Gaussian factor varies smoothly however the
window maps to ln d), and the two sides of the window. The peak is resolved
however narrow it is, and so is the prior's d^2-like growth. About 220
evaluations of the prior density per likelihood call (~0.2 ms). Against
adaptive quadrature: within 1e-8 for a smooth prior, ~2e-5 for bilby's
cosmological priors (their density is interpolated, with kinks); on the
ET-Delta v70 replication (SNR ~500) within 1e-8 of a brute-force distance
integral of the likelihood, where bilby's table is off by -0.4 to +1.6.
"""

import numpy as np
from scipy.special import logsumexp, roots_legendre

from bilby.core.utils import logger

#: the window: where the Gaussian factor is within e^{-K_SIGMA^2/2} of its
#: maximum over the prior's support
K_SIGMA = 10.0
#: chunks of the window, Gauss-Legendre nodes in each and on either side
N_CHUNKS = 10
N_CHUNK = 16
N_SIDE = 32
#: points per piece of the grid that a distance is drawn on
N_SAMPLE_GRID = 400

_NODES_CHUNK = roots_legendre(N_CHUNK)
_NODES_SIDE = roots_legendre(N_SIDE)


def _window(a, b, u_min, u_max):
    """The window [u_lo, u_hi] about the maximum of a u - b u^2 / 2 over
    [u_min, u_max]: K_SIGMA standard deviations about an interior maximum;
    at an edge maximum (slope g' there), down to where the factor has fallen
    by e^{-K_SIGMA^2 / 2} (at most K_SIGMA^2 / 2 / |g'| from the edge)."""
    b = np.maximum(b, 1e-300)
    u_c = np.clip(a / b, u_min, u_max)
    slope = np.abs(a - b * u_c)
    with np.errstate(divide="ignore"):
        half = np.minimum(K_SIGMA / np.sqrt(b), K_SIGMA ** 2 / 2 / slope)
    return np.clip(u_c - half, u_min, u_max), np.clip(u_c + half, u_min, u_max)


def _pieces(a, b, d_ref, d_min, d_max):
    """ln d edges of the pieces, shape (..., N_CHUNKS + 3), increasing: the
    side beyond the window at large d, the window's chunks (equal in u), the
    side at small d."""
    u_lo, u_hi = _window(a, b, d_ref / d_max, d_ref / d_min)
    t = np.linspace(1, 0, N_CHUNKS + 1)
    u = u_lo[..., None] + (u_hi - u_lo)[..., None] * t
    y_min, y_max = np.broadcast_arrays(np.log(d_min), np.log(d_max), u_lo)[:2]
    return np.concatenate(
        [y_min[..., None], np.log(d_ref / u), y_max[..., None]], axis=-1)


def _log_integrand(y, a, b, d_ref, ln_prior):
    """ln of p(d) exp(a u - b u^2 / 2) dd / d(ln d), at y = ln d."""
    d = np.exp(y)
    u = d_ref / d
    return ln_prior(d) + y + a * u - b * u ** 2 / 2


def _nodes(edges):
    """Gauss-Legendre nodes in y and their ln weights on the pieces."""
    ys, lws = [], []
    for k in range(edges.shape[-1] - 1):
        x, w = _NODES_SIDE if k in (0, edges.shape[-1] - 2) else _NODES_CHUNK
        lo, hi = edges[..., k, None], edges[..., k + 1, None]
        half = (hi - lo) / 2
        ys.append(lo + half * (x + 1))
        with np.errstate(divide="ignore"):
            lws.append(np.log(w) + np.log(half))
    return np.concatenate(ys, axis=-1), np.concatenate(lws, axis=-1)


def log_marginal(a, b, d_ref, d_min, d_max, ln_prior):
    """ln int_{d_min}^{d_max} p(d) exp(a u - b u^2 / 2) dd, u = d_ref / d,
    elementwise over arrays a and b. ``ln_prior`` is the log prior density
    of d (vectorised)."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    y, lw = _nodes(_pieces(a, b, d_ref, d_min, d_max))
    return logsumexp(
        _log_integrand(y, a[..., None], b[..., None], d_ref, ln_prior) + lw, axis=-1)


def sample_distance(a, b, d_ref, d_min, d_max, ln_prior, rng=None):
    """One distance from p(d) exp(a u - b u^2 / 2) (scalar a, b): inverse
    CDF on a grid of N_SAMPLE_GRID points in ln d per piece (the CDF by the
    trapezoidal rule, linear within each cell)."""
    rng = np.random.default_rng() if rng is None else rng
    edges = _pieces(np.asarray(a, float), np.asarray(b, float), d_ref, d_min, d_max)
    y = np.unique(np.concatenate([
        np.linspace(edges[k], edges[k + 1], N_SAMPLE_GRID)
        for k in range(len(edges) - 1)]))
    log_f = _log_integrand(y, a, b, d_ref, ln_prior)
    f = np.exp(log_f - log_f.max())
    cdf = np.concatenate([[0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(y))])
    return float(np.exp(np.interp(rng.uniform() * cdf[-1], cdf, y)))


class QuadratureDistanceMarginalization:
    """Mixin for bilby's GravitationalWaveTransient and its subclasses: the
    distance-marginalised likelihood by quadrature (log_marginal) instead of
    bilby's lookup table, and distances for the posterior drawn from the
    exact conditional (sample_distance). Not with phase marginalisation (its
    ln I0 term is not quadratic in u; the higher modes rule it out anyway)."""

    def marginalize_distance(self, priors):
        """Switch distance marginalisation on in a likelihood built without it
        (e.g. one loaded from a pickle), as the constructor would: ``priors``
        (a bilby PriorDict with the luminosity_distance prior, not modified)
        becomes the likelihood's; returns a copy for the sampler, with the
        distance fixed at the reference distance. Relative-binning summary
        data stay valid: the templates scale as 1 / d."""
        if self.distance_marginalization:
            raise ValueError("distance marginalisation is already on")
        if getattr(self, "time_marginalization", False):
            raise NotImplementedError("with time marginalisation")
        self.priors = priors
        self.distance_marginalization = True
        self._ref_dist = float(priors["luminosity_distance"].rescale(0.5))
        self._setup_distance_marginalization()
        self._marginalized_parameters.append("luminosity_distance")
        sampling = priors.copy()
        for key in ("redshift", "comoving_distance"):
            sampling.pop(key, None)
        sampling["luminosity_distance"] = self._ref_dist
        return sampling

    def _setup_distance_marginalization(self, lookup_table=None):
        if lookup_table is not None:
            logger.info("Distance marginalisation by quadrature: the lookup "
                        "table is not used")
        if self.phase_marginalization:
            raise NotImplementedError(
                "distance marginalisation by quadrature without phase "
                "marginalisation only")
        prior = self.priors["luminosity_distance"]
        self._distance_prior = prior
        self._distance_bounds = (float(prior.minimum), float(prior.maximum))
        logger.info(
            f"Distance marginalised by quadrature over {self._distance_bounds[0]:g}-"
            f"{self._distance_bounds[1]:g} Mpc, reference distance {self._ref_dist:g} Mpc")

    def _distance_ln_prior(self, d):
        return self._distance_prior.ln_prob(d)

    def distance_marginalized_likelihood(self, d_inner_h, h_inner_h, parameters=None):
        d_inner_h_ref, h_inner_h_ref = self._setup_rho(
            d_inner_h, h_inner_h, parameters=parameters)
        return log_marginal(np.real(d_inner_h_ref), h_inner_h_ref, self._ref_dist,
                            *self._distance_bounds, self._distance_ln_prior)

    def _rescale_signal(self, signal, new_distance):
        """The polarisations (flat, or per mode as bilby_xG's) from the
        reference distance to ``new_distance``, in place in ``signal``."""
        factor = self._ref_dist / new_distance
        for key, value in signal.items():
            if isinstance(value, dict):
                self._rescale_signal(value, new_distance)
            else:
                signal[key] = value * factor

    def generate_distance_sample_from_marginalized_likelihood(
            self, signal_polarizations=None, parameters=None):
        from bilby.gw.likelihood.base import _fallback_to_parameters
        parameters = _fallback_to_parameters(self, parameters)
        parameters.update(self.get_sky_frame_parameters(parameters))
        if signal_polarizations is None:  # a copy: the generator caches its own
            import copy
            signal_polarizations = copy.deepcopy(
                self.waveform_generator.frequency_domain_strain(parameters))
        d_inner_h, h_inner_h = self._calculate_inner_products(
            signal_polarizations, parameters=parameters)
        d_inner_h_ref, h_inner_h_ref = self._setup_rho(
            d_inner_h, h_inner_h, parameters=parameters)
        from bilby.core.utils import random
        new_distance = sample_distance(
            float(np.real(d_inner_h_ref)), float(h_inner_h_ref), self._ref_dist,
            *self._distance_bounds, self._distance_ln_prior, rng=random.rng)
        self._rescale_signal(signal_polarizations, new_distance)
        return new_distance
