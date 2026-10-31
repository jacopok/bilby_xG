"""The relative-binning likelihood in a frame moving with a constant velocity
(bilby_xG.motion.Center.velocity), on data injected in the barycentric one."""
import numpy as np
import pytest

pytest.importorskip("mlgw_bns")

TC = 1187008882.4 + 1e7


@pytest.fixture(scope="module")
def setup():
    from bilby_xG.injection import SummaryDataInjection
    from bilby_xG.likelihood import (
        RelativeBinningGravitationalWaveTransientNextGenerationModebyMode as Likelihood,
    )
    from bilby_xG.motion import Center, fit_frame_velocity
    from bilby_xG.conversion import barycentre_to_frame_parameters
    from bilby_xG.networks import InterferometerList
    from bilby_xG.source import convert_to_mlgw_bns_parameters, mlgw_bns_individual_modes
    from bilby_xG.waveform_generator import WaveformGenerator

    fmin, fmax, duration = 20.0, 2048.0, 256
    injection = dict(
        chirp_mass=1.1975, mass_ratio=0.9, chi_1=0.01, chi_2=0.0, lambda_1=400.0,
        lambda_2=600.0, luminosity_distance=100.0, theta_jn=0.4, psi=1.0, phase=1.3,
        geocent_time=TC, ra=3.4, dec=-0.4)
    modes = [[2, 2]]
    wfg = WaveformGenerator(
        duration=duration, sampling_frequency=4096,
        frequency_domain_source_model=mlgw_bns_individual_modes,
        parameter_conversion=convert_to_mlgw_bns_parameters,
        waveform_arguments=dict(minimum_frequency=fmin, maximum_frequency=fmax,
                                mode_array=modes))
    start_time = TC + 2 - duration
    data = SummaryDataInjection(wfg, injection, start_time, fmin, fmax, noise=False)

    ifos = InterferometerList(["ET-EMR"])
    frequencies = np.geomspace(fmin, fmax, 4000)
    parameters, _ = convert_to_mlgw_bns_parameters(dict(injection))
    pols = mlgw_bns_individual_modes(frequencies, **parameters, mode_array=modes)["2,2"]
    weights = ((2 * np.pi * frequencies) ** 2 * np.gradient(frequencies)
               * (np.abs(pols["plus"]) ** 2 + np.abs(pols["cross"]) ** 2)
               / ifos[0].power_spectral_density.get_power_spectral_density_array(frequencies))
    velocity = fit_frame_velocity(TC, frequencies, pols["time_to_merger"], weights)
    frame = Center.geocenter(velocity=velocity)
    in_frame = barycentre_to_frame_parameters(injection, velocity)

    def likelihood(center, fiducial):
        return Likelihood(
            interferometers=InterferometerList(["ET-EMR"]), waveform_generator=wfg,
            fiducial_parameters=fiducial, epsilon=0.03, time_reference="geocent",
            mode_array=modes, injection=data, center=center)

    return injection, in_frame, likelihood(None, injection), likelihood(frame, in_frame)


def log_l(likelihood, parameters):
    return float(likelihood.log_likelihood_ratio(parameters=dict(parameters)))


def test_frame_likelihood_at_the_converted_injection(setup):
    injection, in_frame, barycentric, moving = setup
    assert abs(in_frame["chirp_mass"] / injection["chirp_mass"] - 1) > 1e-5
    expected = log_l(barycentric, injection)
    assert expected > 100
    # zero noise: the maximum, <d|d>/2, at the injection, in either frame
    assert abs(log_l(moving, in_frame) - expected) < 1e-3
    # the barycentric parameters in the moving frame are off by the Doppler factor
    assert expected - log_l(moving, injection) > 1


def test_frame_chirp_mass_does_not_follow_the_sky(setup):
    """The best chirp mass (over chirp mass, time and phase) at two sky
    positions far apart, against that of a detector without orbital motion,
    which has no Doppler factor: the moving frame's matches it at both (to
    the curvature of the orbit), the barycentric one is it divided by the
    Doppler factor of each sky position."""
    from scipy.optimize import minimize

    from bilby_xG.motion import frame_doppler_factor

    injection, in_frame, barycentric, moving = setup
    static = setup_static(setup)
    velocity = moving.center.velocity
    for ra in (injection["ra"], injection["ra"] + 2.5):
        best = {}
        for name, likelihood, start in [("barycentric", barycentric, injection),
                                        ("moving", moving, in_frame), ("static", static, in_frame)]:
            def cost(x):
                return -log_l(likelihood, dict(
                    start, ra=ra, chirp_mass=start["chirp_mass"] * (1 + 1e-5 * x[0]),
                    geocent_time=TC + 1e-4 * x[1], phase=start["phase"] + x[2]))

            x = minimize(cost, np.zeros(3), method="Nelder-Mead",
                         options=dict(xatol=1e-4, fatol=1e-6, maxiter=4000)).x
            best[name] = start["chirp_mass"] * (1 + 1e-5 * x[0])
        doppler = frame_doppler_factor(ra, injection["dec"], TC, velocity)
        print(f"ra {ra:.2f}: 1 - D {1 - doppler:+.3e}, best chirp mass / static's: moving "
              f"{best['moving'] / best['static'] - 1:+.2e}, barycentric "
              f"{best['barycentric'] / best['static'] - 1:+.2e}")
        assert abs(1 - doppler) > 3e-5
        assert abs(best["moving"] / best["static"] - 1) < 0.05 * abs(1 - doppler)
        assert abs(best["barycentric"] * doppler / best["static"] - 1) < 0.05 * abs(1 - doppler)


def setup_static(setup):
    """The likelihood of the same data with the detector's orbit held at
    coalescence (the masses of the geocentre's frame then)."""
    _, in_frame, barycentric, _ = setup
    from bilby_xG.likelihood import (
        RelativeBinningGravitationalWaveTransientNextGenerationModebyMode as Likelihood,
    )
    from bilby_xG.networks import InterferometerList

    return Likelihood(
        interferometers=InterferometerList(["ET-EMR"]),
        waveform_generator=barycentric.waveform_generator, fiducial_parameters=in_frame,
        epsilon=0.03, time_reference="geocent", mode_array=[[2, 2]],
        injection=barycentric.injection, orbital_motion=False)


def test_batched_frame_likelihood(setup):
    pytest.importorskip("jax")
    from bilby_xG.batched import BatchedRelativeBinningLikelihood

    injection, in_frame, _, moving = setup
    batched = BatchedRelativeBinningLikelihood(moving, batch_size=4)
    samples = [dict(in_frame), dict(in_frame, ra=in_frame["ra"] + 0.01),
               dict(in_frame, chirp_mass=in_frame["chirp_mass"] * (1 + 3e-6))]
    expected = np.array([log_l(moving, s) for s in samples])
    got = batched.log_likelihood_ratio({key: np.array([s[key] for s in samples])
                                        for key in samples[0]})
    # the JAX surrogate's rounding (see test_batched_likelihood)
    assert np.max(np.abs(got - expected)) < 0.5
