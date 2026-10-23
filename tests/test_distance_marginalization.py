"""Tests for the distance marginalisation by quadrature."""
import numpy as np
import pytest
from scipy.integrate import quad

import bilby
from bilby_xG import distance_marginalization as dm

D_MIN, D_MAX = 25.0, 4000.0


def _smooth_ln_prior(d):
    """p(d) proportional to d^2 on [D_MIN, D_MAX]: smooth, unlike bilby's
    interpolated cosmological priors."""
    return np.log(3 * d ** 2 / (D_MAX ** 3 - D_MIN ** 3))


def _reference(a, b, d_ref, ln_prior):
    """ln of the marginal by adaptive quadrature in ln d."""
    edges = dm._pieces(np.float64(a), np.float64(b), d_ref, D_MIN, D_MAX)
    y = np.linspace(np.log(D_MIN), np.log(D_MAX), 20001)
    y = np.sort(np.concatenate([y, np.linspace(edges[1], edges[-2], 20001)]))
    log_f = dm._log_integrand(y, a, b, d_ref, ln_prior)
    peak = log_f.max()
    value, _ = quad(lambda x: np.exp(dm._log_integrand(x, a, b, d_ref, ln_prior) - peak),
                    np.log(D_MIN), np.log(D_MAX), epsabs=0, epsrel=1e-13, limit=5000,
                    points=sorted({edges[1], edges[-2], y[np.argmax(log_f)]}))
    return np.log(value) + peak


@pytest.mark.parametrize("rho", [0.03, 0.3, 3.0, 30.0, 300.0])
@pytest.mark.parametrize("u_star_fraction", [-1.0, 0.002, 0.05, 0.5, 0.99, 3.0])
def test_log_marginal_matches_adaptive_quadrature(rho, u_star_fraction):
    """SNR rho at the reference distance, the Gaussian's peak u* inside the
    prior (as a fraction of u_max), beyond either edge or at negative u."""
    d_ref = 3000.0
    b = rho ** 2
    a = b * u_star_fraction * d_ref / D_MIN
    value = dm.log_marginal(a, b, d_ref, D_MIN, D_MAX, _smooth_ln_prior)
    assert value == pytest.approx(_reference(a, b, d_ref, _smooth_ln_prior), abs=1e-7, rel=1e-14)


def test_log_marginal_vectorised():
    rng = np.random.default_rng(0)
    b = 10 ** rng.uniform(-2, 4, 50)
    a = b * rng.uniform(-1, 150, 50)
    values = dm.log_marginal(a, b, 3000.0, D_MIN, D_MAX, _smooth_ln_prior)
    singles = [dm.log_marginal(x, y, 3000.0, D_MIN, D_MAX, _smooth_ln_prior)
               for x, y in zip(a, b)]
    np.testing.assert_allclose(values, singles, rtol=0, atol=1e-9)


def test_log_marginal_zero_signal_is_zero():
    """No signal (a = b = 0): the prior's normalisation."""
    value = dm.log_marginal(0.0, 0.0, 3000.0, D_MIN, D_MAX, _smooth_ln_prior)
    assert value == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("rho, d_star", [(0.5, 2000.0), (20.0, 800.0), (500.0, 40.0)])
def test_sample_distance_follows_the_conditional(rho, d_star):
    """Kolmogorov-Smirnov distance between the draws and the conditional's
    CDF (computed by quadrature) below the 1% critical value."""
    d_ref = 3000.0
    b = rho ** 2
    a = b * d_ref / d_star
    rng = np.random.default_rng(1)
    draws = np.sort([dm.sample_distance(a, b, d_ref, D_MIN, D_MAX, _smooth_ln_prior, rng=rng)
                     for _ in range(4000)])
    edges = dm._pieces(np.float64(a), np.float64(b), d_ref, D_MIN, D_MAX)
    y = np.unique(np.concatenate([np.linspace(edges[k], edges[k + 1], 20001)
                                  for k in range(len(edges) - 1)]))
    log_f = dm._log_integrand(y, a, b, d_ref, _smooth_ln_prior)
    f = np.exp(log_f - log_f.max())
    cdf = np.concatenate([[0], np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(y))])
    model = np.interp(np.log(draws), y, cdf / cdf[-1])
    empirical = np.arange(1, len(draws) + 1) / len(draws)
    ks = max(np.abs(model - empirical).max(), np.abs(model - empirical + 1 / len(draws)).max())
    assert ks < 1.63 / np.sqrt(len(draws))


def test_cosmological_prior_within_its_interpolation():
    """bilby's UniformSourceFrame is interpolated (its density has kinks),
    which bounds the agreement at ~1e-5."""
    prior = bilby.gw.prior.UniformSourceFrame(
        minimum=D_MIN, maximum=D_MAX, cosmology="Planck18", name="luminosity_distance")
    d_ref = float(prior.rescale(0.5))
    for rho, fraction in ((0.2, 0.5), (5.0, 0.1), (100.0, 0.9)):
        b = rho ** 2
        a = b * fraction * d_ref / D_MIN
        value = dm.log_marginal(a, b, d_ref, D_MIN, D_MAX, prior.ln_prob)
        assert value == pytest.approx(_reference(a, b, d_ref, prior.ln_prob), abs=5e-5)


def test_likelihood_marginalize_distance():
    """A relative-binning likelihood built without distance marginalisation,
    switched on afterwards: the marginal equals the distance integral of the
    unmarginalised likelihood, and the posterior distances are drawn."""
    pytest.importorskip("lalsimulation")
    import copy
    from test_legacy_likelihood import INJECTION, _likelihood

    likelihood = _likelihood(legacy=True, second_order=False)
    priors = bilby.core.prior.PriorDict(dict(
        luminosity_distance=bilby.core.prior.PowerLaw(
            alpha=2, minimum=5000.0, maximum=60000.0, name="luminosity_distance")))
    marginal = copy.deepcopy(likelihood)
    sampling = marginal.marginalize_distance(priors)
    assert sampling["luminosity_distance"] == marginal._ref_dist
    assert "luminosity_distance" in marginal._marginalized_parameters

    parameters = dict(INJECTION, theta_jn=1.0)
    value = marginal.log_likelihood_ratio(dict(parameters, luminosity_distance=marginal._ref_dist))
    y = np.linspace(np.log(5000.0), np.log(60000.0), 4001)
    log_f = np.array([likelihood.log_likelihood_ratio(dict(parameters, luminosity_distance=d))
                      for d in np.exp(y)]) + priors["luminosity_distance"].ln_prob(np.exp(y)) + y
    f = np.exp(log_f - log_f.max())
    brute = np.log(np.sum((f[1:] + f[:-1]) / 2 * np.diff(y))) + log_f.max()
    assert value == pytest.approx(brute, abs=1e-5)

    bilby.core.utils.random.seed(3)
    draws = [marginal.generate_distance_sample_from_marginalized_likelihood(
        parameters=dict(parameters, luminosity_distance=marginal._ref_dist))
        for _ in range(200)]
    weights = f / f.sum()
    mean = np.sum(weights * np.exp(y))
    std = np.sqrt(np.sum(weights * np.exp(y) ** 2) - mean ** 2)
    assert np.mean(draws) == pytest.approx(mean, abs=4 * std / np.sqrt(200))
