"""Tests for the motion of the detectors in the Solar System."""
import numpy as np
import pytest

from bilby.core.utils import speed_of_light
from bilby_xG.networks import InterferometerList
from bilby_xG.orbit import (
    EarthEphemeris,
    default_ephemeris,
    doppler_factor,
    earth_barycentric_position_velocity,
    hermite_interpolate,
    orbital_time_delay,
    source_direction,
)
from bilby_xG.utils import calculate_time_to_merger_for_any_mode

TC = 1187008882.4


def test_ephemeris_matches_astropy():
    from astropy.coordinates import get_body_barycentric_posvel
    from astropy.time import Time

    times = TC + np.linspace(0, 3e7, 5)
    position, velocity = earth_barycentric_position_velocity(times)
    ap_position, ap_velocity = get_body_barycentric_posvel("earth", Time(times, format="gps"))
    # astropy uses TDB, we neglect TDB - TT (< 2 ms, i.e. < 60 m)
    np.testing.assert_allclose(position, ap_position.xyz.to("m").value.T, rtol=0, atol=100)
    np.testing.assert_allclose(velocity, ap_velocity.xyz.to("m/s").value.T, rtol=0, atol=1e-2)


def test_interpolation_error():
    rng = np.random.default_rng(1)
    times = TC + rng.uniform(-3 * 86400, 86400, 2000)
    exact, _ = earth_barycentric_position_velocity(times)
    assert np.abs(EarthEphemeris().position(times) - exact).max() < 0.05


def test_lazy_extension():
    ephemeris = EarthEphemeris(step=600.0)
    ephemeris.position(np.array([TC]))
    ephemeris.position(np.array([TC + 5 * 86400]))
    ephemeris.position(np.array([TC - 86400]))
    times = np.linspace(TC - 86400, TC + 5 * 86400, 1001)
    assert ephemeris.t0 <= times[0]
    assert len(ephemeris._positions) <= 6 * 144 + 3
    np.testing.assert_allclose(ephemeris.position(times),
                               EarthEphemeris(step=600.0).position(times), rtol=0, atol=1e-4)
    table = ephemeris.table(TC, TC + 3600)
    inside = np.linspace(TC, TC + 3600, 20)
    np.testing.assert_allclose(hermite_interpolate(inside, **table),
                               ephemeris.position(inside), rtol=0, atol=1e-4)


def test_orbital_delay():
    ra, dec = 1.2, -0.3
    ttc = np.geomspace(1e-3, 1e5, 200)
    delay = orbital_time_delay(ra, dec, TC, ttc)
    assert delay.shape == ttc.shape
    _, velocity = earth_barycentric_position_velocity(TC)
    uniform = velocity @ source_direction(ra, dec) * ttc / speed_of_light
    # first order: a uniform motion, whose Doppler factor is doppler_factor
    # (to the ~mm interpolation error, 1e-11 s)
    np.testing.assert_allclose(delay[ttc < 100], uniform[ttc < 100], rtol=1e-5, atol=3e-11)
    np.testing.assert_allclose(doppler_factor(ra, dec, TC),
                               1 - velocity @ source_direction(ra, dec) / speed_of_light)
    # the curvature: centripetal acceleration ~ 6 mm/s^2 for a day
    curvature = np.abs(delay - uniform)[-1]
    assert 1e-3 < curvature < 0.1


def test_response_includes_orbital_delay():
    """Switching on the orbital motion only multiplies the response by
    exp(-2 pi i f delay), the same in every interferometer."""
    ifos = InterferometerList(["ET-EMR"])
    frequencies = np.geomspace(3.0, 2048.0, 300)
    parameters = dict(mass_1=1.45, mass_2=1.3, chi_1=0.0, chi_2=0.0, ra=3.4, dec=-0.4,
                      psi=1.0, geocent_time=TC, luminosity_distance=40.0)
    polarizations = {"plus": np.ones_like(frequencies, dtype=complex),
                     "cross": 1j * np.ones_like(frequencies)}
    ttc = calculate_time_to_merger_for_any_mode(
        frequencies, 1.45, 1.3, 0.0, 0.0, mode=2, safety=1)
    expected = np.exp(-2j * np.pi * frequencies * orbital_time_delay(3.4, -0.4, TC, ttc))
    for ifo in ifos:
        kwargs = dict(waveform_polarizations=polarizations, parameters=parameters,
                      start_time=TC - 1000, frequencies=frequencies)
        on = ifo.get_detector_response_for_frequency_dependent_antenna_response(**kwargs)
        off = ifo.get_detector_response_for_frequency_dependent_antenna_response(
            orbital_motion=False, **kwargs)
        np.testing.assert_allclose(on / off, expected, rtol=0, atol=1e-7)
        shared = {}
        ifo.get_detector_response_for_frequency_dependent_antenna_response(
            shared=shared, orbital_motion=False, **kwargs)
        assert shared[2][3] is None


def test_jax_orbital_delay():
    jnp = pytest.importorskip("jax.numpy")
    import jax

    from bilby_xG.batched import _orbital_delay

    jax.config.update("jax_enable_x64", True)
    ttc = np.geomspace(1e-3, 5e4, 100)
    table = default_ephemeris.table(TC - 86400, TC + 86400)
    delay = _orbital_delay(jnp.float64(1.2), jnp.float64(-0.3), jnp.float64(TC),
                           jnp.asarray(ttc), table)
    np.testing.assert_allclose(np.asarray(delay), orbital_time_delay(1.2, -0.3, TC, ttc),
                               rtol=0, atol=1e-13)


def test_uniform_part_is_a_doppler_rescaling():
    """With the orbital motion, the signal matches the one without it with
    masses and distance multiplied by doppler_factor (and not otherwise)."""
    pytest.importorskip("lalsimulation")
    import bilby
    from scipy.optimize import minimize_scalar

    ifo = InterferometerList(["CE"])[0]
    frequencies = np.geomspace(20.0, 512.0, 100_000)
    weights = np.gradient(frequencies) / \
        ifo.power_spectral_density.get_power_spectral_density_array(frequencies)
    ra, dec, tc = 1.0, 0.5, TC + 1e7

    def strain(factor, orbital_motion, dt=0.0):
        masses = dict(mass_1=1.48 * factor, mass_2=1.27 * factor,
                      luminosity_distance=40.0 * factor)
        pols = bilby.gw.source.binary_black_hole_frequency_sequence(
            frequencies, **masses, a_1=0.0, tilt_1=0.0, phi_12=0.0, a_2=0.0, tilt_2=0.0,
            phi_jl=0.0, theta_jn=0.5, phase=0.0, waveform_approximant="IMRPhenomD",
            reference_frequency=50.0, frequencies=frequencies)
        parameters = dict(masses, chi_1=0.0, chi_2=0.0, ra=ra, dec=dec, psi=0.3,
                          geocent_time=tc + dt)
        return ifo.get_detector_response_for_frequency_dependent_antenna_response(
            pols, parameters, tc - 1e3, frequencies, orbital_motion=orbital_motion)

    def mismatch(a, b):
        return 1 - np.abs(np.sum(weights * np.conj(a) * b)) / np.sqrt(
            np.sum(weights * np.abs(a) ** 2) * np.sum(weights * np.abs(b) ** 2))

    signal = strain(1.0, True)

    def best(factor):
        return minimize_scalar(lambda dt: mismatch(signal, strain(factor, False, dt * 1e-4)),
                               bracket=(-1, 1)).fun

    factor = doppler_factor(ra, dec, tc)
    assert best(factor) < 1e-6
    assert best(1.0) > 1e-3
    assert best(1 / factor) > 1e-3
