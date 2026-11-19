"""Greedy bin selection with eta tuned to an evidence-bias tolerance."""
import json
import os

import numpy as np
import pytest
from scipy.stats import gamma

from bilby_xG.bin_selection import (error_diagnostics, evidence_bias,
                                    nested_sampling_resolution, posterior_range,
                                    sampled_parameter_count)


def test_evidence_bias_is_ln_mean_exp_or_mean_abs():
    rng = np.random.default_rng(0)
    errors = 1e-3 * rng.standard_normal(1000)
    assert evidence_bias(errors) == pytest.approx(np.mean(np.abs(errors)))
    # one-signed errors: ln E[e^eps] > E[eps] = E|eps|
    errors = np.abs(errors) * 100
    assert evidence_bias(errors) == pytest.approx(np.log(np.mean(np.exp(errors))))
    assert evidence_bias(-errors) == pytest.approx(np.mean(errors))


def test_nested_sampling_resolution():
    # Gaussian L(X): d ln L / d ln X = 2 d / D, one dead point per ln X step 1 / n_live
    d = np.array([0.5, 5.5, 50])
    assert np.allclose(nested_sampling_resolution(d, 11, 1000), 2 * d / 11e3)


def test_error_diagnostics_split_at_the_posterior_range():
    dim, n_live = 13, 5000
    d_lo, d_post = posterior_range(dim)
    assert gamma.cdf(d_lo, dim / 2) == pytest.approx(1e-6)
    assert gamma.sf(d_post, dim / 2) == pytest.approx(1e-6)
    # closer to the peak than the posterior: the noise floor, not checked
    d = np.array([0.5 * d_lo, 5.0, d_post, 3e4])
    resolution = nested_sampling_resolution(d, dim, n_live)
    # far from the peak, an error far above the resolution is only a small fraction of d
    errors = np.array([1.0, 0.5 * resolution[1], 0.8 * resolution[2], 300.0])
    ratio, relative = error_diagnostics(d, -errors, dim, n_live)
    assert ratio == pytest.approx(0.8)
    assert relative == pytest.approx(0.01)
    # nothing beyond the posterior
    assert error_diagnostics(d[:3], errors[:3], dim, n_live)[1] == 0


def test_sampled_parameter_count():
    import bilby
    priors = bilby.core.prior.PriorDict(dict(
        a=bilby.core.prior.Uniform(0, 1), b=bilby.core.prior.Sine(), c=1.0,
        d=bilby.core.prior.DeltaFunction(2.0), e=bilby.core.prior.Constraint(0, 1)))
    assert sampled_parameter_count(priors) == 2


@pytest.fixture(scope="module")
def selection(tmp_path_factory):
    pytest.importorskip("mlgw_bns.batched")
    import bilby

    from bilby_xG.bin_selection import select_relative_binning_bins
    from bilby_xG.networks import InterferometerList
    from bilby_xG.source import (MLGW_BNS_MODES, convert_to_mlgw_bns_parameters,
                                 mlgw_bns_individual_modes)
    from bilby_xG.waveform_generator import WaveformGenerator

    fmin, fmax, duration = 30.0, 2048.0, 128
    injection = dict(
        chirp_mass=1.1975, mass_ratio=0.9, chi_1=0.01, chi_2=0.0, lambda_1=400.0,
        lambda_2=600.0, luminosity_distance=400.0, theta_jn=0.4, psi=1.0, phase=1.3,
        geocent_time=1187008882.4, ra=3.4, dec=-0.4)
    wfg = WaveformGenerator(
        duration=duration, sampling_frequency=4096,
        frequency_domain_source_model=mlgw_bns_individual_modes,
        parameter_conversion=convert_to_mlgw_bns_parameters,
        waveform_arguments=dict(minimum_frequency=fmin, maximum_frequency=fmax,
                                mode_array=[list(mode) for mode in MLGW_BNS_MODES]))
    ifos = InterferometerList(["ET-EMR"])
    ifos.set_strain_data_from_zero_noise(4096, duration, injection["geocent_time"] + 2 - duration)
    priors = bilby.core.prior.PriorDict(dict(injection))
    priors["mass_ratio"] = bilby.core.prior.Uniform(0.5, 1)
    priors["lambda_1"] = bilby.core.prior.Uniform(0, 2000)
    priors["lambda_2"] = bilby.core.prior.Uniform(0, 2000)
    priors["luminosity_distance"] = bilby.core.prior.Uniform(200, 800)
    priors["phase"] = bilby.core.prior.Uniform(0, 2 * np.pi)

    def run(name, **kwargs):
        bilby.core.utils.random.seed(1)
        outdir = str(tmp_path_factory.mktemp(name))
        bin_freqs = select_relative_binning_bins(
            ifos, wfg, injection, priors, fmin, fmax, eta=3.0, n_train=4, n_validation=2,
            n_evidence=16, n_scan_draws=1, scan_betas=np.geomspace(1e-4, 1, 5),
            n_grid=5001, npool=4, outdir=outdir, **kwargs)
        with open(os.path.join(outdir, "bin_selection.json")) as f:
            return bin_freqs, json.load(f), outdir

    return dict(tuned=run("tuned", evidence_tolerance=1e-5),
                fixed=run("fixed", evidence_tolerance=None))


def test_tuned_eta_meets_tolerance(selection):
    bin_freqs, summary, outdir = selection["tuned"]
    assert summary["evidence_bias"] <= 1e-5
    assert summary["eta"] == max(h["eta"] for h in summary["eta_history"]
                                 if h["evidence_bias"] <= 1e-5)
    assert summary["n_bins"] == len(bin_freqs) - 1
    assert os.path.exists(os.path.join(outdir, "bin_selection.png"))
    # evidence points spread over the Gamma(D_eff / 2) posterior of d
    d = -np.array(summary["evidence_delta_log_likelihood"])
    # mass_ratio, lambda_1, lambda_2, luminosity_distance, phase
    assert summary["effective_dimension"] == 5
    lo, hi = gamma.ppf([0.01, 0.99], summary["effective_dimension"] / 2)
    assert np.all((d > 0.9 * lo) & (d < 1.1 * hi))


def test_tolerance_none_keeps_eta(selection):
    _, fixed, _ = selection["fixed"]
    _, tuned, _ = selection["tuned"]
    assert fixed["eta"] == 3.0 and len(fixed["eta_history"]) == 1
    # the same first trial, the tuned one goes on to finer bins
    assert tuned["eta_history"][0] == fixed["eta_history"][0]
    assert fixed["evidence_bias"] > 1e-5 and tuned["n_bins"] > fixed["n_bins"]


class _PSD:
    def __init__(self, scale):
        self.scale = scale

    def get_power_spectral_density_array(self, freqs):
        return self.scale * (1 + (10 / freqs) ** 4 + (freqs / 500) ** 2)


class _Ifo:
    def __init__(self, scale):
        self.power_spectral_density = _PSD(scale)


class _Projector:
    """Two detectors, two modes, an ET-like inspiral (SNR ~ 600 up to
    2.7 kHz, as stages_AS_HF_Delta_d1019); ``parameters`` scale the phase
    and give the frequency where the waveform ends (zero above)."""
    m = np.array([2, 3])

    def __init__(self):
        self.interferometers = [_Ifo(1.0), _Ifo(1.3)]

    def project(self, parameters, freqs):
        phase = parameters["phase_scale"] * 1e3 * (freqs / 10) ** (-5 / 3)
        amplitude = np.array([1.0, 0.1])[:, None] * freqs ** (-7 / 6)
        h = amplitude * np.exp(1j * self.m[:, None] * phase / 2)
        h = np.where(freqs <= parameters["cutoff"], h, 0)
        h = np.array([h, 0.8 * h * np.exp(0.3j)])
        return 1.33e3 * h


@pytest.fixture(scope="module")
def cutoff_model():
    from bilby_xG.bin_selection import GridModel
    projector = _Projector()
    freqs = np.geomspace(6.0, 2677.3819307399895, 30_001)
    fiducial = dict(phase_scale=1.0, cutoff=freqs[-1])
    return projector, GridModel(projector, fiducial, freqs)


def test_bin_moments_are_the_bins_trapezoid_sums(cutoff_model):
    _, model = cutoff_model
    f = model.freqs
    lo = np.array([0, 17, 1000, 29_000, 29_999])
    hi = np.array([5, 18, 21_000, 30_000, 30_000])
    A0, A1, B0, B1, B2 = model.bin_moments(lo, hi)
    for j, (a, b) in enumerate(zip(lo, hi)):
        x = f[a:b + 1] - 0.5 * (f[a] + f[b])
        for k, (got_a, got_b) in enumerate([(A0, B0), (A1, B1), (None, B2)]):
            expected_b = np.trapezoid(model.b[..., a:b + 1] * x ** k, x=f[a:b + 1], axis=-1)
            assert np.allclose(got_b[..., j], expected_b, rtol=1e-10, atol=0)
            if got_a is not None:
                expected_a = np.trapezoid(model.a[..., a:b + 1] * x ** k, x=f[a:b + 1], axis=-1)
                assert np.allclose(got_a[..., j], expected_a, rtol=1e-10, atol=0)
    # chunked (bins summed a few at a time) as in one go
    model.MOMENT_CHUNK = 7
    try:
        chunked = model.bin_moments(lo, hi)
    finally:
        del model.MOMENT_CHUNK
    for got, expected in zip(chunked, (A0, A1, B0, B1, B2)):
        assert np.allclose(got, expected, rtol=1e-13, atol=0)


def test_single_cell_bins_are_exact_where_the_waveform_ends(cutoff_model):
    """One bin per grid cell reproduces the grid quadrature, also for a test
    waveform ending inside the last cell (its ratio drops from ~1 to 0):
    prefix-sum moments gave ~1e-4 there (stages_AS_HF_Delta_d1019)."""
    from bilby_xG.bin_selection import GridPoint
    projector, model = cutoff_model
    f = model.freqs
    assert 3e5 < model.norm_d < 5e5
    edges = np.arange(len(f))
    for cutoff in (f[-1] - 1e-3, f[-4] - 1e-3, f[-1]):
        h = projector.project(dict(phase_scale=1 + 5e-6, cutoff=cutoff), f)
        point = GridPoint(model, h)
        assert 1 < -point.delta_log_likelihood < 100
        assert abs(point.total_error(edges)) < 1e-8


def test_select_bins_reports_the_single_cell_floor():
    from bilby_xG.bin_selection import select_bins

    class Model:
        freqs = np.arange(1025.0)

        def bin_moments(self, lo, hi):
            return None

    class Point:
        """Error additive over cells, ~0 except in cell 700."""
        model = Model()

        def __init__(self, bad):
            self.cells = np.full(1024, 1e-12)
            self.cells[700] = bad
            self.prefix = np.concatenate([[0], np.cumsum(self.cells)])

        def bin_errors(self, lo, hi, moments=None):
            assert moments is None
            return self.prefix[hi] - self.prefix[lo]

    edges, floor = select_bins([Point(1e-3), Point(-3e-4)], 1e-2, return_floor=True)
    assert floor == pytest.approx(1e-3)
    assert 700 in edges and 701 in edges
    assert np.array_equal(select_bins([Point(1e-3)], 1e-2), edges)
    assert select_bins([Point(1e-9)], 1e-2, return_floor=True)[1] == 0


def _trials(bias_of_eta):
    def select(eta):
        return np.arange(int(1000 / eta) + 1), 0.0

    return select, lambda edges: bias_of_eta(1000 / (len(edges) - 1))


def test_tune_eta_converges_as_before():
    from bilby_xG.bin_selection import tune_eta
    eta, edges, bias, history, met = tune_eta(*_trials(lambda eta: 1e-4 * eta), 3.0, 1e-5, 8)
    assert met and len(history) == 2 and 5e-6 <= bias <= 1e-5
    assert eta == pytest.approx(3.0 * 0.75e-5 / 3e-4, rel=1e-2)


def test_tune_eta_stops_on_a_stalled_bias_and_keeps_the_cheapest():
    """A bias floor (6e-5) that more bins do not reduce: stop once the bias
    stalls, keep the cheapest trial at the floor, not the last (finest)."""
    from bilby_xG.bin_selection import tune_eta
    from bilby_xG.bin_selection import SAME_BIAS, STALL_STEPS
    eta, edges, bias, history, met = tune_eta(
        *_trials(lambda eta: 3e-5 * eta + 6e-5), 3.0, 1e-5, 8)
    assert not met
    # stopped after STALL_STEPS unresponsive steps, not at the iteration limit
    assert STALL_STEPS < len(history) < 8
    lowest = min(h["evidence_bias"] for h in history)
    cheapest = min(h["n_bins"] for h in history if h["evidence_bias"] <= SAME_BIAS * lowest)
    assert len(edges) - 1 == cheapest < history[-1]["n_bins"]
    assert bias <= SAME_BIAS * lowest


def test_tune_eta_brackets_a_bias_that_drops_in_a_step():
    """A bias that falls from 1e-3 to 1e-8 at eta = 0.3 (the bins resolving
    where test waveforms end): the proportional step alone would jump
    between ~0.02 and ~2; bracketing settles just below 0.3."""
    from bilby_xG.bin_selection import BRACKET_RATIO, tune_eta
    eta, edges, bias, history, met = tune_eta(
        *_trials(lambda eta: 1e-3 if eta > 0.3 else 1e-8), 3.0, 1e-5, 8)
    assert met and bias <= 1e-5
    assert 0.3 / BRACKET_RATIO < eta <= 0.3 * 1.001
    assert eta == max(h["eta"] for h in history if h["evidence_bias"] <= 1e-5)
