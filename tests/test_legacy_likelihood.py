"""Tests for the legacy (bilby-x-g fork) mode-by-mode relative binning."""
import numpy as np
import pytest

import bilby
from bilby.gw.utils import noise_weighted_inner_product
from bilby_xG.injection import SummaryDataInjection
from bilby_xG.likelihood import (
    RelativeBinningGravitationalWaveTransientNextGenerationModebyMode as Likelihood,
    relative_binning_bin_freqs,
)
from bilby_xG.networks import InterferometerList
from bilby_xG.source import lal_binary_black_hole_relative_binning_individual_modes
from bilby_xG.waveform_generator import WaveformGenerator

pytest.importorskip("lalsimulation")

DURATION, SAMPLING_FREQUENCY, FMIN, FMAX = 8, 1024, 20.0, 512.0
MODES = [[2, 2], [2, 1], [3, 3]]
INJECTION = dict(
    mass_1=36.0, mass_2=29.0, a_1=0.3, a_2=0.1, tilt_1=0.5, tilt_2=1.0, phi_12=0.3,
    phi_jl=0.7, chi_1=0.1, chi_2=0.05, luminosity_distance=20000.0, theta_jn=0.8,
    psi=1.0, phase=1.3, geocent_time=1187008882.4, ra=3.4, dec=-0.4, fiducial=0)
START_TIME = INJECTION["geocent_time"] + 2 - DURATION


def _waveform_generator():
    return WaveformGenerator(
        duration=DURATION, sampling_frequency=SAMPLING_FREQUENCY,
        frequency_domain_source_model=lal_binary_black_hole_relative_binning_individual_modes,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=dict(
            waveform_approximant="IMRPhenomXPHM", reference_frequency=20.0,
            minimum_frequency=FMIN, maximum_frequency=FMAX, mode_array=MODES))


def _likelihood(**kwargs):
    wfg = _waveform_generator()
    injection = SummaryDataInjection(wfg, INJECTION, START_TIME, FMIN, FMAX, noise=False,
                                     orbital_motion=False)
    return Likelihood(
        interferometers=InterferometerList(["ET-EMR"]), waveform_generator=wfg,
        fiducial_parameters=INJECTION, epsilon=0.05, time_reference="geocent",
        orbital_motion=False, injection=injection, **kwargs)


def test_legacy_summary_data_follow_the_fork():
    """The fork's summary data, written out as it computed them: the
    fiducial modes on the full grid, its maximum frequency, a1/b1 about each
    bin's sample midpoint and data without the (2, 1) mode."""
    likelihood = _likelihood(legacy=True, second_order=False, summary_data_chunk_size=1000)
    wfg = _waveform_generator()
    frequency_array = wfg.frequency_array
    converted, _ = wfg.parameter_conversion(dict(INJECTION, fiducial=1))
    source = {key: converted[key] for key in wfg.source_parameter_keys}
    polarizations = wfg.frequency_domain_source_model(
        frequency_array, **source, **wfg.waveform_arguments)

    ifo = likelihood.interferometers[0]
    df = frequency_array[1] - frequency_array[0]
    above = frequency_array > FMIN - df
    maximum_frequency = min(
        frequency_array[np.flatnonzero(polarizations[f"{l},{m}"]["plus"][above])[-1]]
        for l, m in MODES)
    assert likelihood.maximum_frequency == maximum_frequency
    expected_bins = relative_binning_bin_freqs(frequency_array, FMIN, maximum_frequency,
                                               chi=1, epsilon=0.05)
    np.testing.assert_array_equal(likelihood.bin_freqs, expected_bins)

    h0 = {}
    for l, m in MODES:
        key = f"{l},{m}"
        h = np.zeros(len(frequency_array), dtype=complex)
        h[above] = ifo.get_detector_response_for_frequency_dependent_antenna_response(
            {key: {p: polarizations[key][p][above] for p in ("plus", "cross")}},
            converted, START_TIME, frequency_array[above], orbital_motion=False)
        h0[key] = h
    data = h0["2,2"] + h0["3,3"]
    mask = ifo.frequency_mask
    frequencies, psd = frequency_array[mask], ifo.power_spectral_density_array[mask]
    edges = np.searchsorted(frequencies, likelihood.bin_freqs)
    summary = likelihood.summary_data[ifo.name]
    for i in (0, len(edges) // 2, len(edges) - 2):
        idxs = slice(edges[i], edges[i + 1])
        f = frequencies[idxs]
        delta = f - (f[0] + f[-1]) / 2
        for key in h0:
            h = h0[key][mask][idxs]
            a1 = noise_weighted_inner_product(h, data[mask][idxs] * delta, psd[idxs], DURATION)
            b1 = noise_weighted_inner_product(h, h0["2,2"][mask][idxs] * delta, psd[idxs],
                                              DURATION)
            np.testing.assert_allclose(summary["a1"][key][i], a1, rtol=1e-10)
            np.testing.assert_allclose(summary["b1"]["2,2"][key][i], b1, rtol=1e-10)
    assert likelihood._legacy_fiducial is None


def test_legacy_needs_the_fork_form():
    with pytest.raises(ValueError, match="second_order"):
        _likelihood(legacy=True)
