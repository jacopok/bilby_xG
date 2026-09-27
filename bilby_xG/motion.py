# Licensed under an MIT style license -- see LICENSE

"""Motion of the detectors in the Solar System.

Each detector is at the barycentric position

.. math::

    \\mathbf{R}_d(t) = \\mathbf{r}_\\oplus(t) + \\mathsf{R}(t)\\,\\mathbf{v}_d,

with :math:`\\mathbf{r}_\\oplus` the barycentric position of the geocentre
(the Earth's orbit, including its monthly wobble about the Earth-Moon
barycentre), :math:`\\mathsf{R}(t)` the Earth's rotation and
:math:`\\mathbf{v}_d` the detector vertex in Earth-fixed coordinates. The
part of the signal emitted at frequency :math:`f`, which reaches the detector
at :math:`t_f = t_c - \\tau(f)`, arrives with the delay

.. math::

    \\Delta t_d(f) = -\\hat{n}\\cdot[\\mathbf{R}_d(t_f) - \\mathbf{C}] / c

with respect to a reference point :math:`\\mathbf{C}`, the :class:`Center`,
where :math:`\\hat{n}` points to the source. The time parameter
(``geocent_time``) is the arrival time at the center: the geocentre at
coalescence by default, as in bilby, or a detector, a weighted average of
the detectors, or any fixed point (see :class:`Center`). Only the positions
of the detectors enter the signal; the geocentre is just a convenient
reference.

For speed the delay is evaluated as the sum of a detector-independent part,
:math:`-\\hat{n}\\cdot[\\mathbf{r}_\\oplus(t_f) - \\mathbf{C}]/c`
(:func:`geocentre_delay`, computed once and shared between the detectors
of a network), and the rotating vertex,
:math:`-\\hat{n}\\cdot\\mathsf{R}(t_f)\\mathbf{v}_d/c`, which
:meth:`bilby_xG.interferometer.Interferometer.frequency_dependent_antenna_response`
already has from the wave frame. The ``earth_rotation_time_delay`` and
``orbital_motion`` options hold the corresponding term at its value at
coalescence.

Frames
------
The sky position follows bilby's convention, in which the Earth's rotation
is ``ra - GMST``: right ascension and declination on the mean equator and
equinox of date. The (ICRS) ephemeris is precessed to the same axes
(IAU 2006, at the coalescence time; the precession over a signal is
negligible).

Frame of the masses
-------------------
To first order in :math:`v/c` a uniform motion multiplies all the
detector-frame masses (and the luminosity distance) by a Doppler factor.
With the orbital motion, they are those of the Solar-System barycentre;
without it, those of a frame comoving with the geocentre at coalescence,
smaller by :func:`doppler_factor` (a :math:`\\sim 10^{-4}` effect). The
choice of center does not change this. The motion enters through the
arrival time only: the corresponding :math:`O(v/c)` amplitude and
:math:`O(v^2/c^2)` phase terms are, for a uniform motion, absorbed by the
same rescaling, and their variation over a signal is negligible.

Ephemeris
---------
Positions come from the ERFA ``epv00`` series (the model behind astropy's
``builtin`` ephemeris), accurate to a few km in absolute terms and much
better in its smooth time variation over a signal. Evaluating the series
costs ~30 us per time, too much at every likelihood evaluation, so
:class:`EarthEphemeris` tabulates positions and velocities on a uniform grid
of nodes, computed lazily over the times actually requested (i.e. per
analysis), and evaluates them by cubic Hermite interpolation. With the
default one-hour node spacing the interpolation error is ~1 cm (tens of
picoseconds of delay), see ``benchmarks/orbital_motion_validation.py``.
"""
import numpy as np
import erfa

from bilby.core.utils import speed_of_light
from bilby_cython.geometry import greenwich_mean_sidereal_time

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


def precession_matrix(gps_time):
    """Rotation from ICRS axes to the mean equator and equinox of date
    (IAU 2006 precession with frame bias), the axes of the sky position."""
    days = gps_time / _DAY
    whole = np.floor(days)
    return erfa.pmat06(_GPS_EPOCH_JD_TT + whole, days - whole)


def sky_direction(ra, dec):
    """Unit vector towards ``(ra, dec)``, in the axes of the sky position."""
    cos_dec = np.cos(dec)
    return np.array([cos_dec * np.cos(ra), cos_dec * np.sin(ra), np.sin(dec)])


def sidereal_rotation(gmst):
    """The Earth's rotation: Earth-fixed to equatorial-of-date axes, shape
    ``np.shape(gmst) + (3, 3)``."""
    gmst = np.asarray(gmst, dtype=float)
    cos, sin = np.cos(gmst), np.sin(gmst)
    zero, one = np.zeros_like(gmst), np.ones_like(gmst)
    return np.stack([np.stack([cos, -sin, zero], -1),
                     np.stack([sin, cos, zero], -1),
                     np.stack([zero, zero, one], -1)], -2)


def greenwich_mean_sidereal_times(gps_times, time):
    """GMST at ``gps_times``: exact at ``time``, advancing at the rate
    over the following day elsewhere, as in
    :func:`bilby_xG.interferometer.compute_wave_frame`."""
    gmst = greenwich_mean_sidereal_time(time)
    rate = (greenwich_mean_sidereal_time(time + _DAY) - gmst) / _DAY
    return gmst + rate * (np.asarray(gps_times, dtype=float) - time)


class Center:
    """The point the arrival time (``geocent_time``) refers to.

    Either a point moving with the Earth, ``offset`` metres from the
    geocentre in Earth-fixed coordinates (the geocentre itself, a detector
    vertex, or a weighted average of vertices), or a fixed barycentric
    ``position``. A point moving with the Earth is taken at the sampled
    coalescence time by default, so that the time parameter is the arrival
    time there, as bilby's geocentre time is, or at a fixed
    ``reference_time``. Build one with the class methods.

    With a center other than the geocentre, sample the time with bilby's
    ``time_reference="geocenter"`` (no conversion) and read ``geocent_time``
    as the arrival time at the center. The choice of center only changes
    the meaning of the time parameter (and not the masses); signals and
    likelihoods are the same up to that relabelling.
    """

    def __init__(self, offset=None, position=None, reference_time=None, label="custom"):
        if (offset is None) == (position is None):
            raise ValueError("give exactly one of offset and position")
        self.offset = None if offset is None else np.asarray(offset, dtype=float)
        self.fixed_position = None if position is None else np.asarray(position, dtype=float)
        self.reference_time = None if reference_time is None else float(reference_time)
        self.label = label

    @classmethod
    def geocenter(cls, reference_time=None):
        """The geocentre (bilby's convention, the default)."""
        return cls(offset=np.zeros(3), reference_time=reference_time, label="geocenter")

    @classmethod
    def interferometer(cls, interferometer, reference_time=None):
        """The vertex of ``interferometer``."""
        return cls(offset=interferometer.geometry.vertex, reference_time=reference_time,
                   label=interferometer.name)

    @classmethod
    def weighted(cls, interferometers, weights, reference_time=None):
        """The ``weights``-weighted average of the vertices, e.g. with the
        optimal SNR squared of each detector as weights. ``weights`` is a
        sequence in the order of ``interferometers`` or a dict by name."""
        if isinstance(weights, dict):
            weights = [weights[ifo.name] for ifo in interferometers]
        weights = np.asarray(weights, dtype=float)
        vertices = np.array([ifo.geometry.vertex for ifo in interferometers])
        return cls(offset=weights @ vertices / weights.sum(), reference_time=reference_time,
                   label="weighted average of " + ", ".join(ifo.name for ifo in interferometers))

    @classmethod
    def fixed(cls, position):
        """A fixed barycentric position (m, ICRS axes), as the center of
        the lgwa-response likelihood."""
        return cls(position=position, label="fixed")

    def position(self, time, ephemeris=None):
        """Barycentric position (m) of the center for coalescence time
        ``time``, in the axes of the sky position at ``time``."""
        if ephemeris is None:
            ephemeris = default_ephemeris
        precession = precession_matrix(time)
        if self.fixed_position is not None:
            return precession @ self.fixed_position
        t = time if self.reference_time is None else self.reference_time
        # the offset in ICRS axes at t
        offset = precession_matrix(t).T @ sidereal_rotation(
            greenwich_mean_sidereal_time(t)) @ self.offset
        return precession @ (ephemeris.position(np.array([t]))[0] + offset)

    def __repr__(self):
        at = "coalescence" if self.reference_time is None else f"GPS {self.reference_time}"
        if self.fixed_position is not None:
            return f"Center.fixed({self.fixed_position.tolist()})"
        return f"Center({self.label}, at {at})"


def resolve_center(center, interferometers=()):
    """A :class:`Center` from ``center``: ``None`` or ``"geocenter"`` (the
    default), the name of one of ``interferometers``, a barycentric position
    (fixed, see :meth:`Center.fixed`), or a :class:`Center`."""
    if center is None or isinstance(center, Center):
        return Center.geocenter() if center is None else center
    if isinstance(center, str):
        if center.lower() in ("geocenter", "geocentre", "geocent"):
            return Center.geocenter()
        for ifo in interferometers:
            if ifo.name == center:
                return Center.interferometer(ifo)
        raise ValueError(f"unknown center {center!r}: not 'geocenter' or an interferometer "
                         f"among {[ifo.name for ifo in interferometers]}")
    return Center.fixed(center)


def geocentre_delay(ra, dec, time, times_to_coalescence, center=None, orbital_motion=True,
                    ephemeris=None):
    """The detector-independent part of the arrival-time delay,
    :math:`-\\hat{n}\\cdot[\\mathbf{r}_\\oplus(t_f) - \\mathbf{C}]/c`.

    Parameters
    ==========
    ra, dec: float
        Source sky position (radians).
    time: float
        Coalescence time at the center (GPS seconds).
    times_to_coalescence: array_like
        Time to coalescence :math:`\\tau` at each frequency; :math:`t_f =`
        ``time`` :math:`- \\tau`.
    center: Center, optional
        Defaults to the geocentre at coalescence, where this vanishes
        without orbital motion.
    orbital_motion: bool
        Follow the geocentre along its orbit; otherwise hold it at its
        position at ``time``.
    ephemeris: EarthEphemeris, optional
        Defaults to :data:`default_ephemeris`.

    Returns
    =======
    array_like
        The delay in seconds, with the shape of ``times_to_coalescence``.
    """
    if ephemeris is None:
        ephemeris = default_ephemeris
    center = resolve_center(center)
    times_to_coalescence = np.asarray(times_to_coalescence, dtype=float)
    direction = sky_direction(ra, dec)
    precession = precession_matrix(time)
    icrs_direction = precession.T @ direction
    if orbital_motion:
        times = time - times_to_coalescence.ravel()
    else:
        times = np.array([time])
    if center.fixed_position is None and center.reference_time is None:
        # the geocentre part of the center is at the coalescence time
        distances = ephemeris.projected_position(np.append(times, time), icrs_direction)
        distances = distances[:-1] - distances[-1]
        distances = distances - direction @ sidereal_rotation(
            greenwich_mean_sidereal_time(time)) @ center.offset
    else:
        distances = (ephemeris.projected_position(times, icrs_direction)
                     - direction @ center.position(time, ephemeris))
    delay = -distances / speed_of_light
    if orbital_motion:
        return delay.reshape(times_to_coalescence.shape)
    return np.full(times_to_coalescence.shape, delay[0])


def detector_position(vertex, gps_times, axes_time=None, ephemeris=None):
    """Barycentric position (m) of an Earth-fixed ``vertex`` at
    ``gps_times``, :math:`\\mathbf{r}_\\oplus(t) + \\mathsf{R}(t)\\mathbf{v}`,
    in the axes of the sky position at ``axes_time`` (default: the first
    time). Shape ``np.shape(gps_times) + (3,)``."""
    if ephemeris is None:
        ephemeris = default_ephemeris
    gps_times = np.asarray(gps_times, dtype=float)
    if axes_time is None:
        axes_time = gps_times.flat[0]
    geocentre = ephemeris.position(gps_times) @ precession_matrix(axes_time).T
    rotation = sidereal_rotation(greenwich_mean_sidereal_times(gps_times, axes_time))
    return geocentre + rotation @ np.asarray(vertex, dtype=float)


def doppler_factor(ra, dec, time):
    """:math:`1 - \\hat{n}\\cdot\\mathbf{v}_\\oplus(t)/c`: the ratio of the
    detector-frame masses (and luminosity distance) inferred without the
    orbital motion to those inferred with it (Solar-System-barycentre
    frame), to first order in :math:`v/c`."""
    _, velocity = earth_barycentric_position_velocity(time)
    return 1 - sky_direction(ra, dec) @ precession_matrix(time) @ velocity / speed_of_light
