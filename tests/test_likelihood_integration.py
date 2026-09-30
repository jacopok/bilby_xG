"""Integration tests for the next-generation likelihoods.

The central correctness check: with all frequency-dependent effects disabled
and a general-relativity propagation model, the response used by
``GravitationalWaveTransientNextGeneration`` must reduce to the standard bilby
detector response, so the two likelihoods agree.
"""
import numpy as np
import pytest

import bilby
from bilby_xG.likelihood import GravitationalWaveTransientNextGeneration
from bilby_xG.networks import InterferometerList

lalsim = pytest.importorskip("lalsimulation")

DURATION = 4.0
SAMPLING_FREQUENCY = 1024.0
INJECTION = dict(
    mass_1=36.0, mass_2=29.0, a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0,
    phi_12=0.0, phi_jl=0.0, luminosity_distance=2000.0, theta_jn=0.4,
    psi=2.659, phase=1.3, geocent_time=1126259642.413, ra=1.375, dec=-1.2108,
    chi_1=0.0, chi_2=0.0,
)


@pytest.fixture(scope="module")
def setup():
    bilby.core.utils.random.seed(42)
    waveform_arguments = dict(
        waveform_approximant="IMRPhenomXP", reference_frequency=50.0,
        minimum_frequency=20.0,
    )
    wfg = bilby.gw.WaveformGenerator(
        duration=DURATION, sampling_frequency=SAMPLING_FREQUENCY,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )
    ifos = InterferometerList(["H1", "L1"])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=SAMPLING_FREQUENCY, duration=DURATION,
        start_time=INJECTION["geocent_time"] - 2,
    )
    ifos.inject_signal(parameters=INJECTION, waveform_generator=wfg)
    return ifos, wfg


def test_nextgen_likelihood_runs(setup):
    ifos, wfg = setup
    like = GravitationalWaveTransientNextGeneration(
        interferometers=ifos, waveform_generator=wfg,
        earth_rotation_beam_patterns=True, earth_rotation_time_delay=True,
        finite_size=True,
    )
    like.parameters.update(INJECTION)
    logl = like.log_likelihood_ratio()
    assert np.isfinite(logl)


def test_gr_reduction_matches_standard(setup):
    """With effects off + GR, NextGen ~= standard GravitationalWaveTransient."""
    ifos, wfg = setup

    standard = bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=wfg,
    )
    nextgen = GravitationalWaveTransientNextGeneration(
        interferometers=ifos, waveform_generator=wfg,
        earth_rotation_beam_patterns=False, earth_rotation_time_delay=False,
        finite_size=False, orbital_motion=False,
    )
    standard.parameters.update(INJECTION)
    nextgen.parameters.update(INJECTION)

    logl_standard = standard.log_likelihood_ratio()
    logl_nextgen = nextgen.log_likelihood_ratio()

    assert np.isfinite(logl_nextgen)
    # Loose tolerance: the two response code paths differ in masking/edge
    # handling but must agree to well within a few percent at the injection.
    assert logl_nextgen == pytest.approx(logl_standard, rel=0.05)


def test_vG_unity_matches_gr(setup):
    """Sampling vG=1 must give the same likelihood as not sampling it."""
    ifos, wfg = setup
    nextgen = GravitationalWaveTransientNextGeneration(
        interferometers=ifos, waveform_generator=wfg,
        earth_rotation_beam_patterns=True, earth_rotation_time_delay=True,
        finite_size=True,
    )
    nextgen.parameters.update(INJECTION)
    logl_gr = nextgen.log_likelihood_ratio()
    nextgen.parameters.update(dict(vG=1.0))
    logl_vg1 = nextgen.log_likelihood_ratio()
    assert logl_vg1 == pytest.approx(logl_gr, rel=1e-10)


def test_center_relabels_the_time(setup):
    """With a detector as the center, the time parameter is the arrival
    time there: the likelihood at the injection's arrival time at H1 is the
    geocentre likelihood at its geocentre time."""
    ifos, wfg = setup
    geocentre = GravitationalWaveTransientNextGeneration(
        interferometers=ifos, waveform_generator=wfg)
    at_h1 = GravitationalWaveTransientNextGeneration(
        interferometers=ifos, waveform_generator=wfg, center="H1")
    assert at_h1.center.label == "H1"
    t_h1 = INJECTION["geocent_time"] + ifos[0].time_delay_from_geocenter(
        INJECTION["ra"], INJECTION["dec"], INJECTION["geocent_time"])
    geocentre.parameters.update(INJECTION)
    at_h1.parameters.update(dict(INJECTION, geocent_time=t_h1))
    # to the ~1e-4 * (t_H1 - t_geo) ~ us by which the center moves, as it
    # is taken at the sampled time
    assert at_h1.log_likelihood_ratio() == pytest.approx(
        geocentre.log_likelihood_ratio(), rel=1e-5)
    at_h1.parameters.update(INJECTION)
    assert abs(at_h1.log_likelihood_ratio() - geocentre.log_likelihood_ratio()) > 1


def test_likelihood_pickled_before_the_motion(setup):
    """A likelihood pickled before orbital_motion and center existed
    evaluates as one without the orbital motion, about the geocentre."""
    import pickle

    ifos, wfg = setup
    old = GravitationalWaveTransientNextGeneration(interferometers=ifos, waveform_generator=wfg)
    del old.orbital_motion, old.center
    old = pickle.loads(pickle.dumps(old))
    without = GravitationalWaveTransientNextGeneration(
        interferometers=ifos, waveform_generator=wfg, orbital_motion=False)
    old.parameters.update(INJECTION)
    without.parameters.update(INJECTION)
    assert old.log_likelihood_ratio() == without.log_likelihood_ratio()
