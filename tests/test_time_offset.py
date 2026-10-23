"""The coalescence time as an offset from a fixed origin, which keeps the
phase of the response exact below the double-precision resolution of a GPS
time (~2.4e-7 s)."""
import numpy as np
import pytest

from bilby_xG.conversion import TIME_OFFSET, TIME_ORIGIN, geocent_time, time_after

ORIGIN = 1187008882.4
#: the spacing of doubles at ORIGIN
ULP = 2.0 ** -22
# below it
SUB_ULP = 5e-8
#: off the peak, where ln L has a slope in time; a multiple of ULP, so that
#: ORIGIN + OFF_PEAK is exact
OFF_PEAK = 839 * ULP


def test_time_after_resolves_sub_ulp_offsets():
    start_time = ORIGIN + 2 - 128
    times = [time_after({TIME_ORIGIN: ORIGIN, TIME_OFFSET: k * SUB_ULP}, start_time)
             for k in range(3)]
    np.testing.assert_allclose(np.diff(times), SUB_ULP, rtol=1e-6)
    # through the absolute time the steps are lost
    absolute = [time_after({"geocent_time": ORIGIN + k * SUB_ULP}, start_time)
                for k in range(3)]
    assert absolute[0] == absolute[1]
    assert geocent_time({TIME_ORIGIN: ORIGIN, TIME_OFFSET: 0.25}) == ORIGIN + 0.25


@pytest.fixture(scope="module")
def setups():
    pytest.importorskip("mlgw_bns")
    from bilby_xG.injection import SummaryDataInjection
    from bilby_xG.likelihood import (
        RelativeBinningGravitationalWaveTransientNextGenerationModebyMode as Likelihood,
    )
    from bilby_xG.networks import InterferometerList
    from bilby_xG.source import (MLGW_BNS_MODES, convert_to_mlgw_bns_parameters,
                                 mlgw_bns_individual_modes)
    from bilby_xG.waveform_generator import WaveformGenerator

    fmin, fmax, duration = 30.0, 2048.0, 128
    injection = dict(
        chirp_mass=1.1975, mass_ratio=0.9, chi_1=0.01, chi_2=0.0, lambda_1=400.0,
        lambda_2=600.0, luminosity_distance=400.0, theta_jn=0.4, psi=1.0, phase=1.3,
        geocent_time=ORIGIN, ra=3.4, dec=-0.4)
    offset_injection = {key: value for key, value in injection.items()
                        if key != "geocent_time"}
    offset_injection.update({TIME_ORIGIN: ORIGIN, TIME_OFFSET: 0.0})
    modes = [list(mode) for mode in MLGW_BNS_MODES]
    wfg = WaveformGenerator(
        duration=duration, sampling_frequency=4096,
        frequency_domain_source_model=mlgw_bns_individual_modes,
        parameter_conversion=convert_to_mlgw_bns_parameters,
        waveform_arguments=dict(minimum_frequency=fmin, maximum_frequency=fmax,
                                mode_array=modes))
    start_time = ORIGIN + 2 - duration
    likelihoods = {}
    for name, parameters in (("absolute", injection), ("offset", offset_injection)):
        data = SummaryDataInjection(wfg, parameters, start_time, fmin, fmax, noise=False)
        likelihoods[name] = Likelihood(
            interferometers=InterferometerList(["ET-EMR"]), waveform_generator=wfg,
            fiducial_parameters=parameters, epsilon=0.03, time_reference="geocent",
            mode_array=modes, injection=data)
    return injection, offset_injection, likelihoods


def _at_offset(offset_injection, offset):
    return dict(offset_injection, **{TIME_OFFSET: offset})


def test_offset_likelihood_matches_absolute_time(setups):
    injection, offset_injection, likelihoods = setups
    # where ORIGIN + delta is exact: elsewhere geocent_time is off by up to
    # ULP / 2, and ln L by ~0.03 at OFF_PEAK at this SNR
    for delta in (0.0, OFF_PEAK, -5033 * ULP):
        absolute = likelihoods["absolute"].log_likelihood_ratio(
            parameters=dict(injection, geocent_time=ORIGIN + delta))
        for likelihood in likelihoods.values():
            # the offset data and fiducial waveform, or the absolute ones
            offset = likelihood.log_likelihood_ratio(
                parameters=_at_offset(offset_injection, delta))
            np.testing.assert_allclose(offset, absolute, rtol=1e-6, atol=1e-6)


def test_offset_likelihood_is_smooth_below_the_resolution_of_gps_times(setups):
    injection, offset_injection, likelihoods = setups
    likelihood = likelihoods["offset"]
    delta = OFF_PEAK
    values = [likelihood.log_likelihood_ratio(
        parameters=_at_offset(offset_injection, delta + k * SUB_ULP)) for k in range(3)]
    steps = np.diff(values)
    assert steps[0] != 0
    np.testing.assert_allclose(steps[1], steps[0], rtol=1e-3)
    # the same steps in geocent_time are lost
    absolute = [likelihood.log_likelihood_ratio(
        parameters=dict(injection, geocent_time=ORIGIN + delta + k * SUB_ULP))
        for k in range(2)]
    assert absolute[0] == absolute[1]


def test_batched_likelihood_with_offset(setups):
    pytest.importorskip("jax")
    from bilby_xG.batched import BatchedRelativeBinningLikelihood

    injection, offset_injection, likelihoods = setups
    batched = BatchedRelativeBinningLikelihood(likelihoods["offset"], batch_size=4)
    deltas = OFF_PEAK + SUB_ULP * np.arange(4)
    got = batched.log_likelihood_ratio(
        {key: np.full(4, value) for key, value in offset_injection.items()}
        | {TIME_OFFSET: deltas})
    absolute = batched.log_likelihood_ratio(
        {key: np.full(4, value) for key, value in injection.items()}
        | {"geocent_time": ORIGIN + deltas})
    np.testing.assert_allclose(got[0], absolute[0], atol=1e-6)
    # resolved below the spacing of GPS times, smoothly
    steps = np.diff(got)
    assert steps[0] != 0
    np.testing.assert_allclose(steps, steps[0], rtol=1e-3)


def test_point_finder_keeps_the_time_origin_exact():
    from bilby_xG.bin_selection import PointFinder

    finder = PointFinder.__new__(PointFinder)
    finder.fiducial = {TIME_ORIGIN: ORIGIN, TIME_OFFSET: 0.0, "ra": 1.0, "fiducial": 0}
    finder.keys = [TIME_ORIGIN, TIME_OFFSET, "ra"]
    for beta in np.geomspace(1e-7, 1, 30):
        point = finder.combine({TIME_OFFSET: 0.05, "ra": 2.0}, beta)
        assert point[TIME_ORIGIN] == ORIGIN
        assert point[TIME_OFFSET] == pytest.approx(0.05 * beta)
