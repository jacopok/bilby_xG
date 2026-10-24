"""The bin-selection grid model with data other than the fiducial waveform:
a signal plus Gaussian noise drawn on the grid itself."""
import numpy as np
import pytest

from bilby_xG.bin_selection import grid_noise


def test_grid_noise_has_the_variance_of_full_resolution_noise():
    rng = np.random.default_rng(0)
    freqs = np.geomspace(5, 500, 400)
    psd = np.vstack([1 + (20 / freqs) ** 4, 2 + 0 * freqs])
    h = np.vstack([np.exp(2j * np.pi * freqs / 37) / freqs, (1 + 1j) / np.sqrt(freqs)])
    norm = 4 * np.trapezoid(np.abs(h) ** 2 / psd, x=freqs).sum()
    inner = [4 * np.trapezoid(np.conj(grid_noise(psd, freqs, rng)) * h / psd, x=freqs).sum()
             for _ in range(4000)]
    # Re <n|h> ~ N(0, <h|h>)
    assert np.var(np.real(inner)) == pytest.approx(norm, rel=0.08)
    assert np.var(np.imag(inner)) == pytest.approx(norm, rel=0.08)
    assert abs(np.mean(np.real(inner))) < 0.1 * np.sqrt(norm)


@pytest.fixture(scope="module")
def noisy_model():
    pytest.importorskip("mlgw_bns")
    from bilby_xG.bin_selection import GridModel, Projector
    from bilby_xG.networks import InterferometerList
    from bilby_xG.source import (MLGW_BNS_MODES, convert_to_mlgw_bns_parameters,
                                 mlgw_bns_individual_modes)
    from bilby_xG.waveform_generator import WaveformGenerator

    fmin, fmax, duration = 30.0, 2048.0, 128
    signal = dict(
        chirp_mass=1.1975, mass_ratio=0.9, chi_1=0.01, chi_2=0.0, lambda_1=400.0,
        lambda_2=600.0, luminosity_distance=400.0, theta_jn=0.4, psi=1.0, phase=1.3,
        geocent_time=1187008882.4, ra=3.4, dec=-0.4, fiducial=0)
    wfg = WaveformGenerator(
        duration=duration, sampling_frequency=4096,
        frequency_domain_source_model=mlgw_bns_individual_modes,
        parameter_conversion=convert_to_mlgw_bns_parameters,
        waveform_arguments=dict(minimum_frequency=fmin, maximum_frequency=fmax,
                                mode_array=[list(mode) for mode in MLGW_BNS_MODES]))
    ifos = InterferometerList(["ET-EMR"])
    ifos.set_strain_data_from_zero_noise(4096, duration, signal["geocent_time"] + 2 - duration)
    projector = Projector(ifos, wfg)
    freqs = np.geomspace(fmin, fmax, 3001)
    clean = GridModel(projector, signal, freqs)
    data = clean.d + grid_noise(clean.psd, freqs, np.random.default_rng(1))
    # the fiducial is not the signal
    fiducial = dict(signal, chirp_mass=1.19751, theta_jn=0.45)
    return projector, GridModel(projector, fiducial, freqs, data=data), signal


def test_relative_binning_likelihood_is_exact_plus_total_error(noisy_model):
    from bilby_xG.bin_selection import GridPoint

    projector, model, signal = noisy_model
    edges = np.unique(np.round(np.geomspace(1, len(model.freqs), 41)).astype(int) - 1)
    h = projector.project(dict(signal, mass_ratio=0.89), model.freqs)
    point = GridPoint(model, h)
    exact = point.E[-1] - point.E[0]
    relative_binning = model.relative_binning_log_likelihood(edges, h[..., edges])
    assert relative_binning == pytest.approx(exact + point.total_error(edges), abs=1e-9)
    # the noise makes the relative-binning error linear near the fiducial
    assert relative_binning != pytest.approx(exact, abs=1e-6)


def test_delta_log_likelihood_from_the_maximum(noisy_model):
    from bilby_xG.bin_selection import GridPoint, delta_log_likelihood

    projector, model, signal = noisy_model
    h_max = projector.project(signal, model.freqs)
    model.set_maximum(h_max)
    assert delta_log_likelihood(model, h_max) == pytest.approx(0, abs=1e-9)
    h = projector.project(dict(signal, psi=1.01), model.freqs)
    expected = GridPoint(model, h).E[-1] - GridPoint(model, h_max).E[-1]
    assert delta_log_likelihood(model, h) == pytest.approx(expected, rel=1e-8)


def test_relative_binning_overlaps(noisy_model):
    projector, model, signal = noisy_model
    edges = np.unique(np.round(np.geomspace(1, len(model.freqs), 41)).astype(int) - 1)
    h = projector.project(dict(signal, mass_ratio=0.89), model.freqs[edges])
    dh, hh = model.relative_binning_overlaps(edges, h)
    assert model.relative_binning_log_likelihood(edges, h) == pytest.approx(
        dh.real - hh / 2, rel=1e-12)
    # the likelihood of the rescaled and rotated waveform is the maximized one
    best = h * abs(dh) / hh * dh / abs(dh)
    assert model.relative_binning_log_likelihood(edges, best) == pytest.approx(
        abs(dh) ** 2 / (2 * hh), rel=1e-9)
