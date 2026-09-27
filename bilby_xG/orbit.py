# Licensed under an MIT style license -- see LICENSE

"""Motion of the detectors in the Solar System.

The frequency-dependent response in :mod:`bilby_xG.interferometer` places
the detectors on a rotating Earth whose centre is at rest. This module adds
the motion of the geocentre itself with respect to the Solar-System
barycentre (the Earth's orbit, including its monthly wobble about the
Earth-Moon barycentre), as an extra, detector-independent arrival-time delay

.. math::

    \\Delta t(f) = -\\hat{n} \\cdot [\\mathbf{r}_\\oplus(t_c - \\tau(f))
                   - \\mathbf{r}_\\oplus(t_c)] / c,

where :math:`\\hat{n}` points to the source, :math:`t_c` is the geocentre
coalescence time and :math:`\\tau(f)` the time to coalescence from
frequency :math:`f`. The geocentre time is therefore still the arrival time
at the geocentre, and the delay vanishes at coalescence.

Frame of the masses
-------------------
To first order in :math:`v/c` the delay is a uniform Doppler shift. Without
it the waveform is effectively defined in a frame comoving with the
geocentre at :math:`t_c`; with it, in the frame of the Solar-System
barycentre. The detector-frame masses (and luminosity distance) of the two
conventions differ by the factor returned by :func:`doppler_factor`,

.. math::

    M_\\text{no motion} = M_\\text{barycentre} (1 - \\hat{n}\\cdot\\mathbf{v}_\\oplus(t_c)/c),

a :math:`\\sim 10^{-4}` effect. The departure of the motion from a uniform
one over the signal (its curvature) is what cannot be absorbed this way.

The motion enters through the arrival time only, as the Earth's rotation
does. The changes it also makes to the amplitude (first order in
:math:`v/c`) and the second-order phase terms are, for a uniform motion,
absorbed by the same rescaling of the masses and distance, and their
variation over a signal is negligible.

Ephemeris
---------
Positions come from the ERFA ``epv00`` series (the model behind astropy's
``builtin`` ephemeris), which is accurate to a few km in absolute terms
and much better in its smooth time variation over a signal. Evaluating the
series costs ~30 us per time, too much at every likelihood evaluation, so
:class:`EarthEphemeris` tabulates positions and velocities on a uniform grid
of nodes, computed lazily over the times actually requested (i.e. per
analysis), and evaluates them by cubic Hermite interpolation. With the
default one-hour node spacing the interpolation error is ~1 mm (a few
picoseconds of delay), see ``benchmarks/orbital_motion_validation.py``.
"""
import numpy as np
import erfa

from bilby.core.utils import speed_of_light

__author__ = ["Jacopo Tissino"]

#: Astronomical unit in metres (IAU 2012).
ASTRONOMICAL_UNIT = 149597870700.0
_DAY = 86400.0
#: Julian date (TT) of the GPS epoch, 1980-01-06T00:00:00 UTC; TT - GPS is
#: a constant 51.184 s. The difference between TT and TDB (<2 ms, periodic
#: over a year) is neglected: it shifts the argument of the ephemeris by a
#: nearly-constant amount, a sub-100-m change in the position.
_GPS_EPOCH_JD_TT = 2444244.5 + 51.184 / _DAY


def earth_barycentric_position_velocity(gps_times):
    """Barycentric position (m) and velocity (m/s) of the geocentre, ICRS
    axes, from the ERFA ``epv00`` series (exact, not interpolated).

    Returns two arrays of shape ``np.shape(gps_times) + (3,)``.
    """
    gps_times = np.asarray(gps_times, dtype=float)
    days = gps_times / _DAY
    whole = np.floor(days)
    # split the Julian date for precision
    _, pvb = erfa.epv00(_GPS_EPOCH_JD_TT + whole, days - whole)
    return (pvb["p"] * ASTRONOMICAL_UNIT,
            pvb["v"] * (ASTRONOMICAL_UNIT / _DAY))


def hermite_interpolate(times, t0, step, positions, velocities, xp=np):
    """Cubic Hermite interpolation on the uniform grid ``t0 + k * step``.

    ``positions`` and ``velocities`` have shape ``(n_nodes, 3)``; times
    outside the grid are extrapolated from the first/last interval. ``xp``
    is the array module (``numpy`` or ``jax.numpy``). Returns shape
    ``times.shape + (3,)``.
    """
    s = (times - t0) / step
    k = xp.clip(xp.floor(s).astype(int), 0, positions.shape[0] - 2)
    s = (s - k)[..., None]
    s2 = s * s
    s3 = s2 * s
    h00 = 2 * s3 - 3 * s2 + 1
    h10 = s3 - 2 * s2 + s
    h01 = -2 * s3 + 3 * s2
    h11 = s3 - s2
    return (h00 * positions[k] + h10 * step * velocities[k]
            + h01 * positions[k + 1] + h11 * step * velocities[k + 1])


class EarthEphemeris:
    """Lazily-tabulated barycentric position of the geocentre.

    Nodes sit at ``k * step`` GPS seconds; the table always spans a
    contiguous range of nodes and is extended (never recomputed) when a time
    outside it is requested, so it only ever covers the times an analysis
    uses.

    Parameters
    ==========
    step: float
        Node spacing in seconds.
    """

    def __init__(self, step=3600.0):
        self.step = float(step)
        self._first = None
        self._positions = np.empty((0, 3))
        self._velocities = np.empty((0, 3))

    @property
    def t0(self):
        """GPS time of the first node."""
        return self._first * self.step

    def _nodes(self, first, last):
        return earth_barycentric_position_velocity(
            np.arange(first, last + 1) * self.step)

    def ensure(self, t_min, t_max):
        """Tabulate the nodes needed to interpolate over ``[t_min, t_max]``."""
        first = int(np.floor(t_min / self.step))
        last = int(np.floor(t_max / self.step)) + 1
        if self._first is None:
            self._positions, self._velocities = self._nodes(first, last)
            self._first = first
            return
        current_last = self._first + len(self._positions) - 1
        if first < self._first:
            p, v = self._nodes(first, self._first - 1)
            self._positions = np.concatenate([p, self._positions])
            self._velocities = np.concatenate([v, self._velocities])
            self._first = first
        if last > current_last:
            p, v = self._nodes(current_last + 1, last)
            self._positions = np.concatenate([self._positions, p])
            self._velocities = np.concatenate([self._velocities, v])

    def position(self, gps_times):
        """Interpolated barycentric position (m) at ``gps_times``, shape
        ``np.shape(gps_times) + (3,)``."""
        gps_times = np.asarray(gps_times, dtype=float)
        self.ensure(np.min(gps_times), np.max(gps_times))
        return hermite_interpolate(gps_times, self.t0, self.step,
                                   self._positions, self._velocities)

    def projected_position(self, gps_times, direction):
        """Interpolated ``position(gps_times) @ direction`` (m), cheaper
        than :meth:`position` since only the nodes spanning ``gps_times``
        are projected and the interpolation is one-dimensional."""
        gps_times = np.asarray(gps_times, dtype=float)
        t_min, t_max = np.min(gps_times), np.max(gps_times)
        self.ensure(t_min, t_max)
        first = int(np.floor(t_min / self.step)) - self._first
        last = int(np.floor(t_max / self.step)) + 1 - self._first
        nodes = slice(first, last + 1)
        return hermite_interpolate(
            gps_times, (self._first + first) * self.step, self.step,
            (self._positions[nodes] @ direction)[:, None],
            (self._velocities[nodes] @ direction)[:, None])[..., 0]

    def table(self, t_min, t_max):
        """The node table covering ``[t_min, t_max]``, as a dict with keys
        ``t0``, ``step``, ``positions``, ``velocities`` for
        :func:`hermite_interpolate` (e.g. to evaluate it in JAX)."""
        self.ensure(t_min, t_max)
        first = int(np.floor(t_min / self.step)) - self._first
        last = int(np.floor(t_max / self.step)) + 1 - self._first
        return dict(t0=(self._first + first) * self.step, step=self.step,
                    positions=self._positions[first:last + 1].copy(),
                    velocities=self._velocities[first:last + 1].copy())


#: The ephemeris shared by all interferometers (per process).
default_ephemeris = EarthEphemeris()


def source_direction(ra, dec):
    """Unit vector towards ``(ra, dec)`` in ICRS axes."""
    cos_dec = np.cos(dec)
    return np.array([cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)])


def orbital_time_delay(ra, dec, time, times_to_coalescence, ephemeris=None):
    """Arrival-time delay due to the geocentre's motion in the Solar System.

    Parameters
    ==========
    ra, dec: float
        Source sky position (radians).
    time: float
        Geocentre coalescence time (GPS seconds), where the delay is zero.
    times_to_coalescence: array_like
        Time to coalescence at each frequency.
    ephemeris: EarthEphemeris, optional
        Defaults to :data:`default_ephemeris`.

    Returns
    =======
    array_like
        :math:`-\\hat{n}\\cdot[\\mathbf{r}(t - \\tau) - \\mathbf{r}(t)]/c`, in
        seconds, with the shape of ``times_to_coalescence``.
    """
    if ephemeris is None:
        ephemeris = default_ephemeris
    times_to_coalescence = np.asarray(times_to_coalescence, dtype=float)
    distances = ephemeris.projected_position(
        np.append(time - times_to_coalescence.ravel(), time), source_direction(ra, dec))
    delay = -(distances[:-1] - distances[-1]) / speed_of_light
    return delay.reshape(times_to_coalescence.shape)


def doppler_factor(ra, dec, time):
    """:math:`1 - \\hat{n}\\cdot\\mathbf{v}_\\oplus(t)/c`: the ratio of the
    detector-frame masses (and luminosity distance) inferred without the
    orbital motion to those inferred with it (Solar-System-barycentre
    frame), to first order in :math:`v/c`."""
    _, velocity = earth_barycentric_position_velocity(time)
    return 1 - velocity @ source_direction(ra, dec) / speed_of_light
