"""Tests for injections as relative-binning summary data."""
import numpy as np
import pytest

import bilby
from bilby.gw.utils import noise_weighted_inner_product
from bilby_xG.injection import SummaryDataInjection, inject_zero_noise_chunked
from bilby_xG.likelihood import (
    RelativeBinningGravitationalWaveTransientNextGenerationModebyMode as Likelihood,
)
from bilby_xG.networks import InterferometerList
from bilby_xG.source import lal_binary_black_hole_relative_binning_individual_modes
from bilby_xG.waveform_generator import WaveformGenerator

pytest.importorskip("lalsimulation")

DURATION, SAMPLING_FREQUENCY, FMIN, FMAX = 8, 1024, 20.0, 512.0
INJECTION = dict(
    mass_1=36.0, mass_2=29.0, a_1=0.3, a_2=0.1, tilt_1=0.5, tilt_2=1.0, phi_12=0.3,
    phi_jl=0.7, chi_1=0.1, chi_2=0.05, luminosity_distance=20000.0, theta_jn=0.8,
    psi=1.0, phase=1.3, geocent_time=1187008882.4, ra=3.4, dec=-0.4, fiducial=0)
# off the fiducial point, so the a1 terms matter
OTHER = dict(INJECTION, mass_1=36.0144, phase=1.6, ra=3.402)
START_TIME = INJECTION["geocent_time"] + 2 - DURATION


def _waveform_generator():
    return WaveformGenerator(
        duration=DURATION, sampling_frequency=SAMPLING_FREQUENCY,
        frequency_domain_source_model=lal_binary_black_hole_relative_binning_individual_modes,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(
            waveform_approximant="IMRPhenomXPHM", reference_frequency=20.0,
            minimum_frequency=FMIN, maximum_frequency=FMAX,
            mode_array=[[2, 2], [2, 1], [3, 3]]))


def _likelihood(noise=True, seed=1, chunk_size=2 ** 20):
    wfg = _waveform_generator()
    injection = SummaryDataInjection(wfg, INJECTION, START_TIME, FMIN, FMAX,
                                     noise=noise, seed=seed)
    return Likelihood(
        interferometers=InterferometerList(["ET-EMR"]), waveform_generator=wfg,
        fiducial_parameters=INJECTION, epsilon=0.05, time_reference="geocent",
        summary_data_chunk_size=chunk_size, injection=injection)


def _log_likelihood_ratio(likelihood, parameters):
    likelihood.parameters.update(parameters)
    return likelihood.log_likelihood_ratio()


def test_zero_noise_matches_the_full_strain():
    ifos, wfg = InterferometerList(["ET-EMR"]), _waveform_generator()
    inject_zero_noise_chunked(ifos, wfg, INJECTION, START_TIME, FMIN, FMAX, progress=False)
    full = Likelihood(interferometers=ifos, waveform_generator=wfg,
                      fiducial_parameters=INJECTION, epsilon=0.05, time_reference="geocent")
    summary = _likelihood(noise=False)
    assert np.array_equal(full.bin_freqs, summary.bin_freqs)
    for ifo, summary_ifo in zip(ifos, summary.interferometers):
        assert summary_ifo.strain_data._frequency_domain_strain is None
        for kind in ("a0", "a1"):
            for mode, expected in full.summary_data[ifo.name][kind].items():
                np.testing.assert_allclose(
                    summary.summary_data[ifo.name][kind][mode], expected,
                    rtol=0, atol=1e-10 * np.abs(expected).max())
    np.testing.assert_allclose(summary.noise_log_likelihood(),
                               full.noise_log_likelihood(), rtol=1e-8)
    for parameters in (INJECTION, OTHER):
        np.testing.assert_allclose(_log_likelihood_ratio(summary, parameters),
                                   _log_likelihood_ratio(full, parameters), rtol=1e-8)


def test_noise_realisation_is_fixed_by_the_seed_alone():
    a, b = _likelihood(seed=3), _likelihood(seed=3, chunk_size=1000)
    c = _likelihood(seed=4)
    for ifo in a.interferometers:
        for mode, expected in a.summary_data[ifo.name]["a0"].items():
            np.testing.assert_allclose(b.summary_data[ifo.name]["a0"][mode], expected,
                                       rtol=1e-10, atol=0)
            assert not np.allclose(c.summary_data[ifo.name]["a0"][mode], expected)


def test_noise_statistics():
    """Over noise realisations, ln L(x) = <x|s> - <x|x>/2 + Re<x|n> with
    Re<x|n> ~ N(0, <x|x>) and Cov(Re<x|n>, Re<y|n>) = Re<x|y>; and
    ln L(s) + ln L_noise = -<n|n>/2, which the noise ln L leaves out."""
    likelihood = _likelihood()
    exact = SummaryDataInjection(_waveform_generator(), INJECTION, START_TIME, FMIN, FMAX,
                                 noise=False)
    other = SummaryDataInjection(_waveform_generator(), OTHER, START_TIME, FMIN, FMAX,
                                 noise=False)
    freqs = likelihood.waveform_generator.frequency_array
    freqs = freqs[(freqs >= FMIN) & (freqs <= FMAX)]
    ss = hh = hs = 0.0
    for ifo in likelihood.interferometers:
        psd = ifo.power_spectral_density.get_power_spectral_density_array(freqs)
        s, h = exact.signal(ifo, freqs), other.signal(ifo, freqs)
        ss += noise_weighted_inner_product(s, s, psd, DURATION).real
        hh += noise_weighted_inner_product(h, h, psd, DURATION).real
        hs += noise_weighted_inner_product(h, s, psd, DURATION).real

    draws = []
    for seed in range(100):
        likelihood.injection.seed = seed
        likelihood.compute_summary_data()
        draws.append([_log_likelihood_ratio(likelihood, INJECTION),
                      _log_likelihood_ratio(likelihood, OTHER),
                      likelihood.noise_log_likelihood()])
    truth, off, noise = np.array(draws).T
    n = len(truth)
    # 4 sigma: means have standard error sqrt(var/n), variances about var sqrt(2/n)
    assert abs(truth.mean() - ss / 2) < 4 * np.sqrt(ss / n)
    assert abs(off.mean() - (hs - hh / 2)) < 4 * np.sqrt(hh / n)
    assert abs(truth.var() / ss - 1) < 4 * np.sqrt(2 / n)
    assert abs(off.var() / hh - 1) < 4 * np.sqrt(2 / n)
    assert abs(np.cov(truth, off)[0, 1] / hs - 1) < 4 * np.sqrt(2 / n)
    np.testing.assert_allclose(truth + noise, 0, atol=1e-6 * ss)


def test_refuses_what_needs_the_full_data():
    wfg = _waveform_generator()
    injection = SummaryDataInjection(wfg, INJECTION, START_TIME, FMIN, FMAX)
    with pytest.raises(ValueError, match="time marginalization"):
        Likelihood(interferometers=InterferometerList(["ET-EMR"]), waveform_generator=wfg,
                   fiducial_parameters=INJECTION, time_marginalization=True,
                   priors=bilby.gw.prior.BBHPriorDict(), injection=injection)
