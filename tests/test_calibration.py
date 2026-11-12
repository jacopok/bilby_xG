"""Calibration uncertainty (bilby's CubicSpline model) in the relative-binning
likelihoods: the spline at fixed frequencies against bilby's, the calibrated
ln L against the exact one on the full grid, and the batched likelihood
against the numpy one."""
import numpy as np
import pytest

from bilby.gw.detector.calibration import CubicSpline
from bilby.gw.utils import noise_weighted_inner_product

from bilby_xG.calibration import CalibrationAtFrequencies

pytest.importorskip("mlgw_bns")

FMIN, FMAX, DURATION = 30.0, 2048.0, 128
N_NODES = 10


def nodes(name, rng, amplitude=0.05, phase=0.05):
    return {**{f"recalib_{name}_amplitude_{k}": amplitude * rng.normal() for k in range(N_NODES)},
            **{f"recalib_{name}_phase_{k}": phase * rng.normal() for k in range(N_NODES)}}


def test_spline_matches_bilby():
    model = CubicSpline("recalib_H1_", FMIN, FMAX, N_NODES)
    f = np.geomspace(FMIN, FMAX, 777)
    parameters = nodes("H1", np.random.default_rng(3))
    expected = model.get_calibration_factor(f, **parameters)
    got = CalibrationAtFrequencies(model, f).factor(parameters)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)
    # arrays of parameter sets: one column each
    many = {key: np.array([value, 2 * value]) for key, value in parameters.items()}
    got = CalibrationAtFrequencies(model, f).factor(many)
    np.testing.assert_allclose(got[:, 0], expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        got[:, 1], model.get_calibration_factor(f, **{k: 2 * v for k, v in parameters.items()}),
        rtol=1e-12, atol=1e-14)


def test_nominal_and_partial_parameters():
    calibration = CalibrationAtFrequencies(CubicSpline("recalib_H1_", FMIN, FMAX, N_NODES),
                                           np.geomspace(FMIN, FMAX, 50))
    assert calibration.factor(dict(chirp_mass=1.2)) is None
    partial = nodes("H1", np.random.default_rng(0))
    partial.pop("recalib_H1_phase_3")
    with pytest.raises(KeyError):
        calibration.factor(partial)


@pytest.fixture(scope="module")
def setup():
    from bilby_xG.injection import SummaryDataInjection
    from bilby_xG.likelihood import (
        RelativeBinningGravitationalWaveTransientNextGenerationModebyMode as Likelihood,
    )
    from bilby_xG.networks import InterferometerList
    from bilby_xG.source import (MLGW_BNS_MODES, convert_to_mlgw_bns_parameters,
                                 mlgw_bns_individual_modes)
    from bilby_xG.waveform_generator import WaveformGenerator

    injection = dict(
        chirp_mass=1.1975, mass_ratio=0.9, chi_1=0.01, chi_2=0.0, lambda_1=400.0,
        lambda_2=600.0, luminosity_distance=60.0, theta_jn=0.6, psi=1.0, phase=1.3,
        geocent_time=1187008882.4, ra=3.4, dec=-0.4)
    modes = [list(mode) for mode in MLGW_BNS_MODES]
    wfg = WaveformGenerator(
        duration=DURATION, sampling_frequency=4096,
        frequency_domain_source_model=mlgw_bns_individual_modes,
        parameter_conversion=convert_to_mlgw_bns_parameters,
        waveform_arguments=dict(minimum_frequency=FMIN, maximum_frequency=FMAX,
                                mode_array=modes))
    data = SummaryDataInjection(wfg, injection, injection["geocent_time"] + 2 - DURATION,
                                FMIN, FMAX, noise=False)
    ifos = InterferometerList(["H1", "L1"])
    for ifo in ifos:
        ifo.calibration_model = CubicSpline(f"recalib_{ifo.name}_", FMIN, FMAX, N_NODES)
    likelihood = Likelihood(
        interferometers=ifos, waveform_generator=wfg, fiducial_parameters=injection,
        epsilon=0.03, time_reference="geocent", mode_array=modes, injection=data)
    return injection, data, likelihood


def exact_log_likelihood_ratio(data, likelihood, calibration):
    """Re <s|C s> - <C s|C s> / 2 on the full grid, s the injected signal (the
    data, and the waveform at the injection), C bilby's calibration factor."""
    total = 0.0
    for ifo in likelihood.interferometers:
        f = np.arange(int(np.ceil(FMIN * DURATION)), int(FMAX * DURATION) + 1) / DURATION
        s = data.signal(ifo, f)
        c = ifo.calibration_model.get_calibration_factor(f, **calibration)
        psd = ifo.power_spectral_density.get_power_spectral_density_array(f)
        h = c * s
        total += (noise_weighted_inner_product(s, h, psd, DURATION).real
                  - noise_weighted_inner_product(h, h, psd, DURATION).real / 2)
    return total


def test_calibrated_likelihood_matches_full_grid(setup):
    injection, data, likelihood = setup
    rng = np.random.default_rng(7)
    nominal = likelihood.log_likelihood_ratio(parameters=dict(injection))
    calibration = {**nodes("H1", rng), **nodes("L1", rng, amplitude=0.03, phase=0.03)}
    calibrated = likelihood.log_likelihood_ratio(parameters=dict(injection, **calibration))
    exact = exact_log_likelihood_ratio(data, likelihood, calibration)
    zero = {key: 0.0 for key in calibration}
    # the nominal calibration: no parameters, or zero nodes
    np.testing.assert_allclose(
        likelihood.log_likelihood_ratio(parameters=dict(injection, **zero)), nominal, rtol=1e-12)
    np.testing.assert_allclose(nominal, exact_log_likelihood_ratio(data, likelihood, zero),
                               rtol=1e-6)
    print(f"ln L nominal {nominal:.6f}, calibrated {calibrated:.6f}, exact {exact:.6f}")
    # the calibration moves ln L by ~1.6 nats (5% and 3% at SNR ~50);
    # relative binning follows it to within a small fraction of that
    assert nominal - exact > 1
    assert abs(calibrated - exact) < 1e-2 * (nominal - exact)


def test_batched_calibration_matches_numpy(setup):
    pytest.importorskip("jax")
    from bilby_xG.batched import BatchedRelativeBinningLikelihood

    injection, _, likelihood = setup
    batched = BatchedRelativeBinningLikelihood(likelihood, batch_size=4)
    rng = np.random.default_rng(11)
    samples = [dict(injection, **nodes("H1", rng), **nodes("L1", rng)) for _ in range(6)]
    expected = np.array([likelihood.log_likelihood_ratio(parameters=dict(s)) for s in samples])
    got = batched.log_likelihood_ratio({key: np.array([s[key] for s in samples])
                                        for key in samples[0]})
    # the batched surrogate's rounding (test_batched_likelihood), at SNR ~50
    assert np.max(np.abs(got - expected)) < 0.5
    # the calibration's own effect is the same in both
    nominal = batched.log_likelihood_ratio(dict(injection))
    nominal_numpy = likelihood.log_likelihood_ratio(parameters=dict(injection))
    np.testing.assert_allclose(got - nominal, expected - nominal_numpy, atol=0.05)
    # without calibration parameters: the nominal calibration
    single = batched.log_likelihood_ratio(dict(samples[0]))
    np.testing.assert_allclose(single, got[0], rtol=1e-6)
