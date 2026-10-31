# Licensed under an MIT style license -- see LICENSE

"""Parameter-conversion helpers.

Adds support for sampling in ``log10_luminosity_distance`` (base-10 log of the
luminosity distance in Mpc) on top of bilby's standard CBC conversions. Pass
these converters explicitly, e.g.::

    bilby.gw.WaveformGenerator(
        ...,
        parameter_conversion=bilby_xG.conversion.convert_to_lal_binary_black_hole_parameters,
    )

Also the coalescence time as an offset from a fixed origin
(:func:`time_after`, :func:`geocent_time`).
"""
import numpy as np
from bilby.gw.conversion import (
    convert_to_lal_binary_black_hole_parameters as _convert_bbh,
    convert_to_lal_binary_neutron_star_parameters as _convert_bns,
)

__author__ = ["Pratyusava Baral <pbaral@uwm.edu>", "Soichiro Morisaki"]

#: The coalescence time (s) relative to ``time_origin`` (GPS s): sampled in
#: place of ``geocent_time``, with ``time_origin`` a fixed parameter (a
#: number or a ``DeltaFunction`` in the prior dict), see :func:`time_after`.
TIME_OFFSET = "geocent_time_offset"
TIME_ORIGIN = "time_origin"


def geocent_time(parameters):
    """The coalescence time ``geocent_time`` (GPS s), from ``time_origin``
    and ``geocent_time_offset`` when the offset is given.

    Rounded to double precision (2**-22 s ~ 2.4e-7 s for GPS times between
    2**30 and 2**31 s): only for quantities varying slowly with time, such
    as the Earth's orientation and position; phases need
    :func:`time_after`."""
    if TIME_OFFSET in parameters:
        return parameters[TIME_ORIGIN] + parameters[TIME_OFFSET]
    return parameters["geocent_time"]


def time_after(parameters, start_time):
    """``geocent_time - start_time``, at full precision when the time is
    given as ``geocent_time_offset`` from ``time_origin``.

    A GPS time ~1e9 s is resolved in double precision only to ~2.4e-7 s,
    which at the SNRs of next-generation detectors is a sizeable fraction
    of the width of the time posterior: through ``geocent_time`` the
    likelihood is a staircase in time. Here the large difference
    ``time_origin - start_time`` is taken first (exactly, for nearby times)
    and the offset added to it, so the absolute time is never formed."""
    if TIME_OFFSET in parameters:
        return (parameters[TIME_ORIGIN] - start_time) + parameters[TIME_OFFSET]
    return parameters["geocent_time"] - start_time


def _add_log10_luminosity_distance(parameters):
    """Return a copy with ``luminosity_distance`` set from its base-10 log.

    Only acts when ``log10_luminosity_distance`` is present and an explicit
    ``luminosity_distance`` is not; otherwise the parameters are returned
    unchanged.
    """
    if ("log10_luminosity_distance" in parameters
            and "luminosity_distance" not in parameters):
        parameters = parameters.copy()
        parameters["luminosity_distance"] = \
            10 ** parameters["log10_luminosity_distance"]
    return parameters


def _add_aligned_spin_components(converted, added_keys):
    """Add ``chi_i = a_i cos(tilt_i)`` when absent (precessing spins), which
    the frequency-dependent detector response uses for its time-frequency
    map."""
    for i in (1, 2):
        if f"chi_{i}" not in converted and f"a_{i}" in converted:
            converted[f"chi_{i}"] = converted[f"a_{i}"] * np.cos(converted.get(f"tilt_{i}", 0.0))
            added_keys = added_keys + [f"chi_{i}"]
    return converted, added_keys


def convert_to_lal_binary_black_hole_parameters(parameters):
    """As bilby's BBH converter, but also accepts ``log10_luminosity_distance``,
    and gives ``chi_i`` for precessing spins."""
    return _add_aligned_spin_components(*_convert_bbh(_add_log10_luminosity_distance(parameters)))


def convert_to_lal_binary_neutron_star_parameters(parameters):
    """As bilby's BNS converter, but also accepts ``log10_luminosity_distance``,
    and gives ``chi_i`` for precessing spins."""
    return _add_aligned_spin_components(*_convert_bns(_add_log10_luminosity_distance(parameters)))


#: the parameters a uniform motion of the observer rescales (to first order
#: in v/c, by the Doppler factor), wherever they are given
FRAME_SCALED_PARAMETERS = ("chirp_mass", "total_mass", "mass_1", "mass_2",
                           "luminosity_distance")


def _frame_factor(parameters, velocity, time=None):
    from .motion import frame_doppler_factor

    if time is None:
        # the Earth's axes and the frame's Doppler factor vary negligibly
        # over a posterior's times: one time for all the samples
        time = float(np.mean(geocent_time(parameters)))
    return frame_doppler_factor(np.asarray(parameters["ra"]), np.asarray(parameters["dec"]),
                                time, velocity)


def barycentre_to_frame_parameters(parameters, velocity, time=None):
    """``parameters`` with the detector-frame masses and luminosity distance
    (those of :data:`FRAME_SCALED_PARAMETERS` present) of the Solar-System
    barycentre replaced by those of a frame moving with ``velocity`` (m/s,
    ICRS axes; :attr:`bilby_xG.motion.Center.velocity`): multiplied by
    :func:`bilby_xG.motion.frame_doppler_factor` at the sky position.
    ``time`` (GPS s) defaults to the samples' mean coalescence
    time. Parameters given at a reference frequency (a precessing
    waveform's phase) are left as they are: they depend on the waveform
    model. A dict, or anything indexed like one (a pandas DataFrame), is
    copied."""
    factor = _frame_factor(parameters, velocity, time)
    out = parameters.copy()
    for key in FRAME_SCALED_PARAMETERS:
        if key in out:
            out[key] = out[key] * factor
    return out


def frame_to_barycentre_parameters(parameters, velocity, time=None):
    """The inverse of :func:`barycentre_to_frame_parameters`."""
    factor = _frame_factor(parameters, velocity, time)
    out = parameters.copy()
    for key in FRAME_SCALED_PARAMETERS:
        if key in out:
            out[key] = out[key] / factor
    return out
