"""Tests for the chunked zero-noise injection."""
import numpy as np
import pytest

import bilby
from bilby_xG.injection import inject_zero_noise_chunked
from bilby_xG.networks import InterferometerList
from bilby_xG.source import (
    binary_neutron_star_individual_modes_frequency_sequence,
    lal_binary_black_hole_relative_binning_individual_modes,
)
from bilby_xG.waveform_generator import WaveformGenerator

pytest.importorskip("lalsimulation")

DURATION, SAMPLING_FREQUENCY, FMIN = 64, 1024, 20.0
INJECTION = dict(
    mass_1=1.6, mass_2=1.4, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0,
    phi_12=0.0, phi_jl=0.0, chi_1=0.0, chi_2=0.0, lambda_1=0.0, lambda_2=0.0,
    luminosity_distance=100.0, theta_jn=0.4, psi=1.0, phase=1.3,
    geocent_time=1187008882.4, ra=3.4, dec=-0.4)
WAVEFORM_ARGUMENTS = dict(
    waveform_approximant="IMRPhenomXPHM", reference_frequency=20.0,
    minimum_frequency=FMIN, maximum_frequency=SAMPLING_FREQUENCY / 2,
    mode_array=[[2, 2], [2, 1], [3, 3], [4, 4]])
START_TIME = INJECTION["geocent_time"] + 2 - DURATION


def _waveform_generator():
    return WaveformGenerator(
        duration=DURATION, sampling_frequency=SAMPLING_FREQUENCY,
        frequency_domain_source_model=lal_binary_black_hole_relative_binning_individual_modes,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=WAVEFORM_ARGUMENTS)


def _inject(chunk_size):
    ifos = InterferometerList(["ET-EMR"])
    snr = inject_zero_noise_chunked(
        ifos, _waveform_generator(), INJECTION, START_TIME, FMIN,
        SAMPLING_FREQUENCY / 2, chunk_size=chunk_size, progress=False)
    return ifos, snr


def test_chunking_does_not_change_the_injection():
    small, snr_small = _inject(1000)
    whole, snr_whole = _inject(2 ** 30)
    for a, b in zip(small, whole):
        np.testing.assert_array_equal(a.frequency_domain_strain, b.frequency_domain_strain)
    assert snr_small == snr_whole > 0


def test_matches_all_modes_frequency_sequence_response():
    ifos, _ = _inject(4096)
    wfg = _waveform_generator()
    freqs = wfg.frequency_array
    mask = freqs > FMIN - 1 / DURATION
    params, _ = wfg.parameter_conversion(dict(INJECTION))
    pols = binary_neutron_star_individual_modes_frequency_sequence(
        frequency_array=freqs[mask], frequencies=freqs[mask],
        **WAVEFORM_ARGUMENTS, **params)
    for ifo in ifos:
        expected = np.zeros_like(freqs, dtype=complex)
        expected[mask] = ifo.get_detector_response_for_frequency_dependent_antenna_response(
            waveform_polarizations=pols, parameters=params, start_time=START_TIME,
            frequencies=freqs[mask], earth_rotation_time_delay=True,
            earth_rotation_beam_patterns=True, finite_size=True)
        np.testing.assert_allclose(ifo.frequency_domain_strain, expected,
                                   rtol=0, atol=1e-10 * np.abs(expected).max())
        assert np.all(ifo.frequency_domain_strain[~mask] == 0)


def test_frequency_sequence_keeps_m1_modes():
    """Regression: the |m| grouping used to loop over m = 2, 3, 4 only and
    silently dropped (2, 1)."""
    freqs = np.arange(FMIN, 256, 1 / 16)
    params = {k: v for k, v in INJECTION.items()
              if k not in ("psi", "geocent_time", "ra", "dec", "chi_1", "chi_2")}
    pols = binary_neutron_star_individual_modes_frequency_sequence(
        frequency_array=freqs, frequencies=freqs, **WAVEFORM_ARGUMENTS, **params)
    assert set(pols) == {"1", "2", "3", "4"}
    rb = lal_binary_black_hole_relative_binning_individual_modes(
        freqs, **{k: v for k, v in params.items() if not k.startswith("lambda")},
        fiducial=0, frequency_bin_edges=freqs, **WAVEFORM_ARGUMENTS)
    for m, key in [("1", "2,1"), ("2", "2,2"), ("3", "3,3"), ("4", "4,4")]:
        assert np.abs(pols[m]["plus"]).max() > 0
        np.testing.assert_allclose(pols[m]["plus"], rb[key]["plus"],
                                   rtol=0, atol=1e-10 * np.abs(rb[key]["plus"]).max())
