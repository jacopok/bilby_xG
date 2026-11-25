"""The time at a (2,2) frequency as the time parameter of the mlgw_bns sources."""
import numpy as np
import pytest

pytest.importorskip("mlgw_bns")

from bilby_xG.source import (  # noqa: E402
    MLGW_BNS_MODES, convert_to_mlgw_bns_parameters, convert_to_mlgw_bns_precessing_parameters,
    mlgw_bns_has_precessing_times, mlgw_bns_individual_modes,
    mlgw_bns_precessing_individual_modes)

COMMON = dict(chirp_mass=1.1975, mass_ratio=0.9, lambda_1=400.0, lambda_2=600.0,
              luminosity_distance=40.0, theta_jn=0.4, phase=1.3)
ALIGNED = COMMON | dict(chi_1=0.01, chi_2=0.0)
PRECESSING = COMMON | dict(a_1=0.3, a_2=0.1, tilt_1=1.2, tilt_2=2.0, phi_12=1.0, phi_jl=0.7)
F0, OFFSET = 100.0, 1.9
REFERENCE = dict(time_reference_frequency=F0, time_reference_offset=OFFSET)
FREQS = np.sort(np.append(np.geomspace(20.0, 1500.0, 80), F0))
MODES = [list(mode) for mode in MLGW_BNS_MODES]


def source(precessing, parameters=None, freqs=FREQS, **kwargs):
    import inspect
    function, convert, parameters_ = (
        (mlgw_bns_precessing_individual_modes, convert_to_mlgw_bns_precessing_parameters,
         PRECESSING) if precessing
        else (mlgw_bns_individual_modes, convert_to_mlgw_bns_parameters, ALIGNED))
    converted, _ = convert(dict(parameters or parameters_))
    names = [n for n, p in inspect.signature(function).parameters.items()
             if n not in ("frequency_array", "kwargs")]
    return function(freqs, **{n: converted[n] for n in names}, mode_array=MODES,
                    minimum_frequency=20.0, reference_frequency=20.0, **kwargs)


@pytest.mark.parametrize("precessing", [False, True])
def test_time_reference_is_a_shift(precessing):
    if precessing:
        pytest.importorskip("jax")
        if not mlgw_bns_has_precessing_times():
            pytest.skip("mlgw_bns without precessing times")
    merger, reference = source(precessing), source(precessing, **REFERENCE)
    at = np.searchsorted(FREQS, F0)
    tau = merger["2,2"]["time_to_merger"][at]
    assert 1 < tau < 4
    # the (2,2) emits F0 at geocent_time - offset
    np.testing.assert_allclose(reference["2,2"]["time_to_merger"][at], OFFSET, atol=1e-6)
    ratios = []
    for mode in merger:
        a, b = merger[mode], reference[mode]
        np.testing.assert_allclose(b["time_to_merger"], a["time_to_merger"] + OFFSET - tau,
                                   atol=2e-6)
        ratio = b["plus"] / a["plus"] * np.exp(-2j * np.pi * FREQS * (OFFSET - tau))
        # a rotation of m times one orbital phase (none if it is fixed at
        # reference_frequency by the precessing source)
        np.testing.assert_allclose(ratio, ratio[0], atol=1e-2)
        ratios.append((int(mode.split(",")[1]), ratio[0]))
    if precessing:
        for _, ratio in ratios:
            np.testing.assert_allclose(ratio, 1.0, atol=1e-2)


def test_precessing_time_to_merger_is_close_to_2pn():
    """The multipoles' own maps, against the 2PN the response used before."""
    pytest.importorskip("jax")
    if not mlgw_bns_has_precessing_times():
        pytest.skip("mlgw_bns without precessing times")
    from bilby_xG.utils import calculate_time_to_merger_for_any_mode
    band = FREQS < 100
    converted, _ = convert_to_mlgw_bns_precessing_parameters(dict(PRECESSING))
    for mode, pols in source(True).items():
        post_newtonian = calculate_time_to_merger_for_any_mode(
            FREQS[band], converted["mass_1"], converted["mass_2"], converted["chi_1"],
            converted["chi_2"], mode=int(mode.split(",")[1]), safety=1)
        np.testing.assert_allclose(pols["time_to_merger"][band], post_newtonian, rtol=4e-2)


def test_teobresums_at_the_same_reference():
    pytest.importorskip("EOBRun_module")
    pytest.importorskip("jax")
    from bilby_xG.teobresums_mlgw import teobresums_mlgw_individual_modes
    converted, _ = convert_to_mlgw_bns_parameters(dict(ALIGNED))
    names = ("M", "q", "chi1z", "chi2z", "LambdaAl2", "LambdaBl2", "distance", "inclination",
             "coalescence_angle")
    teob = teobresums_mlgw_individual_modes(
        FREQS, **{n: converted[n] for n in names}, mode_array=MODES, minimum_frequency=20.0,
        **REFERENCE)
    surrogate = source(False, **REFERENCE)
    band = (FREQS > 30) & (FREQS < 800)
    at = np.searchsorted(FREQS, F0)
    assert abs(teob["2,2"]["time_to_merger"][at] - OFFSET) < 1e-6
    relative = np.angle(teob["2,2"]["plus"][band] / surrogate["2,2"]["plus"][band])
    assert np.max(abs(relative)) < 0.05


@pytest.mark.parametrize("precessing", [False, True])
def test_batched_likelihood_with_a_time_reference(precessing, monkeypatch):
    pytest.importorskip("jax")
    if precessing and not mlgw_bns_has_precessing_times():
        pytest.skip("mlgw_bns without precessing times")
    from bilby_xG.batched import BatchedRelativeBinningLikelihood
    from bilby_xG.injection import SummaryDataInjection
    from bilby_xG.likelihood import (
        RelativeBinningGravitationalWaveTransientNextGenerationModebyMode as Likelihood)
    from bilby_xG.networks import InterferometerList
    from bilby_xG.waveform_generator import WaveformGenerator

    fmin, fmax, duration = 30.0, 2048.0, 128
    injection = (PRECESSING if precessing else ALIGNED) | dict(
        luminosity_distance=400.0, psi=1.0, geocent_time=1187008882.4, ra=3.4, dec=-0.4)
    wfg = WaveformGenerator(
        duration=duration, sampling_frequency=4096,
        frequency_domain_source_model=(mlgw_bns_precessing_individual_modes if precessing
                                       else mlgw_bns_individual_modes),
        parameter_conversion=(convert_to_mlgw_bns_precessing_parameters if precessing
                              else convert_to_mlgw_bns_parameters),
        waveform_arguments=dict(minimum_frequency=fmin, maximum_frequency=fmax,
                                reference_frequency=fmin, mode_array=MODES, **REFERENCE))
    data = SummaryDataInjection(wfg, injection, injection["geocent_time"] + 2 - duration,
                                fmin, fmax, noise=False)
    likelihood = Likelihood(
        interferometers=InterferometerList(["ET-EMR"]), waveform_generator=wfg,
        fiducial_parameters=injection, epsilon=0.03, time_reference="geocent",
        mode_array=MODES, injection=data)
    batched = BatchedRelativeBinningLikelihood(likelihood, batch_size=2)
    samples = [dict(injection), dict(injection, lambda_1=430.0, theta_jn=0.43,
                                     geocent_time=injection["geocent_time"] + 1e-4)]
    expected = [likelihood.log_likelihood_ratio(parameters=dict(s)) for s in samples]
    got = batched.log_likelihood_ratio({k: np.array([s[k] for s in samples]) for k in injection})
    assert expected[0] > 100
    assert abs(got[0] - expected[0]) < 0.05
    assert abs(got[1] - expected[1]) < 0.5
