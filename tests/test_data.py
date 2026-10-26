"""Tests for the data realisation and the re-binned summary data."""
import numpy as np
import pytest

from bilby_xG.data import DataRealisation, fine_likelihood, overlaps, rebin
from bilby_xG.likelihood import (
    RelativeBinningGravitationalWaveTransientNextGenerationModebyMode as Likelihood,
    _log_likelihood_ratio_at)
from bilby_xG.networks import InterferometerList

from test_summary_data_injection import (FMAX, FMIN, INJECTION, OTHER, START_TIME,
                                         _waveform_generator)

pytest.importorskip("lalsimulation")

# a fiducial away from the injection, as a search's would be
FIDUCIAL = dict(INJECTION, mass_1=36.05, ra=3.45, psi=1.2)


def _data(wfg, seed=5):
    return DataRealisation(wfg, INJECTION, START_TIME, FMIN, FMAX, seed=seed)


def test_noise_realisation_does_not_depend_on_the_frequencies_asked():
    wfg = _waveform_generator()
    ifo = InterferometerList(["ET-EMR"])[0]
    data = _data(wfg)
    data.setup_interferometers([ifo])
    grid = np.arange(int(FMIN * wfg.duration), int(FMAX * wfg.duration)) / wfg.duration
    whole = data.noise_at(ifo, grid)
    parts = np.concatenate([data.noise_at(ifo, part) for part in np.array_split(grid, 7)])
    np.testing.assert_array_equal(whole, parts)
    shuffled = np.random.default_rng(0).permutation(len(grid))
    np.testing.assert_array_equal(data.noise_at(ifo, grid[shuffled]), whole[shuffled])
    # E[|n|^2] = S T / 2
    psd = ifo.power_spectral_density.get_power_spectral_density_array(grid)
    assert abs(np.mean(np.abs(whole) ** 2 / (psd * wfg.duration / 2)) - 1) < 0.1
    assert not np.array_equal(whole, _data(wfg, seed=6).noise_at(ifo, grid))


def test_rebinned_summary_data_match_a_direct_pass():
    fine = fine_likelihood(InterferometerList(["ET-EMR"]), _waveform_generator(),
                           _data(_waveform_generator()), INJECTION, 1000, epsilon=0.05)
    coarse = rebin(fine, FIDUCIAL, minimum_frequency=FMIN, number_of_bins=60)
    direct = Likelihood(
        interferometers=InterferometerList(["ET-EMR"]), waveform_generator=_waveform_generator(),
        fiducial_parameters=FIDUCIAL, bin_freqs=coarse.bin_freqs, time_reference="geocent",
        injection=_data(_waveform_generator()), epsilon=0.05)
    np.testing.assert_allclose(direct.bin_freqs, coarse.bin_freqs)
    np.testing.assert_allclose(coarse.noise_log_likelihood(), direct.noise_log_likelihood())
    for parameters in (INJECTION, OTHER, FIDUCIAL):
        expected = _log_likelihood_ratio_at(direct, parameters)
        assert abs(_log_likelihood_ratio_at(coarse, parameters) - expected) < 1e-3 * abs(expected) + 1e-2
        d_h, h_h = overlaps(coarse, parameters)
        d_h_direct, h_h_direct = overlaps(direct, parameters)
        np.testing.assert_allclose(h_h, h_h_direct, rtol=1e-4)
        assert abs(d_h - d_h_direct) < 1e-4 * abs(d_h_direct) + 1e-3
