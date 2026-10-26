# Licensed under an MIT style license -- see LICENSE

"""The maximum-likelihood point from the data and a search trigger alone,
the fiducial of a relative-binning analysis that does not know the signal.

What the analysis knows, and the order in which it learns it:

- the data (:class:`bilby_xG.data.DataRealisation`, or any object with its
  interface);
- a trigger, as a search would report it: the detector-frame chirp mass, a
  mass ratio, the arrival time at the first interferometer and the network
  SNR; no sky, spins or tides (:func:`emulated_trigger` makes one from an
  injection, with errors);
- one pass over the data with very many bins around a fiducial built from
  the trigger (:func:`initial_fiducial`, :func:`bilby_xG.data.fine_likelihood`),
  from which every later relative-binning likelihood is re-binned without
  touching the data again (:func:`bilby_xG.data.rebin`);
- a global search (differential evolution over the chirp mass, mass ratio,
  orientation, sky and arrival time) on the relative-binning likelihood
  maximized over the amplitude and an overall phase, from
  ``search_f_min`` with ``search_bins`` bins, from ``n_searches`` seeds,
  the phase and distance then fitted; then each result and its 7 images
  under the symmetries of the detector's response (nearly degenerate sky
  modes, see :func:`sky_images`), those close on the sky merged, refined:
  local maximization over all the sampled parameters (the spins Cartesian)
  with ``bins`` bins, re-binned around each new maximum until the gain is
  below ``tolerance``, from each of the minimum frequencies
  ``continuation`` in turn (the sky modes are only told apart at the
  lowest, by the Earth's rotation). The best is the maximum-likelihood
  point theta_hat; all are kept, as candidate modes.

The parameters are bilby's, with the coalescence time as
``geocent_time_offset`` from ``time_origin`` (the trigger's time) and
precessing spins (``a_i``, ``tilt_i``, ``phi_12``, ``phi_jl``) or aligned
ones (``chi_i``).
"""

import dataclasses
import json
import multiprocessing
import os
import time

import bilby
import numpy as np
from bilby.core.prior import Constraint, DeltaFunction, Prior
from bilby.core.utils import logger
from scipy.optimize import differential_evolution, minimize, minimize_scalar

from .bin_selection import GridModel, Projector
from .conversion import TIME_OFFSET, TIME_ORIGIN
from .data import overlaps, rebin
from .likelihood import _log_likelihood_ratio_at
from .motion import doppler_factor, sky_direction

#: tidal deformabilities of the first fiducial (the trigger has none)
INITIAL_LAMBDA = 500.0
#: inclination of the first fiducial: not face on, where the modes with
#: |m| != 2 vanish and the ratios to them are 0 / 0
INITIAL_THETA_JN = 1.0
#: angles taken modulo their prior range
PERIODIC = {"phase", "psi", "ra", "phi_12", "phi_jl"}
#: the arrival-time shift from the trigger, searched instead of geocent_time_offset
TIME_SHIFT = "dt"
#: the precessing spins in the local searches, Cartesian, as a (sin tilt cos phi,
#: sin tilt sin phi, cos tilt) with phi = phi_jl for the first and
#: phi_jl + phi_12 for the second: smooth through a = 0, where the tilts and
#: azimuths are degenerate and a search in them stalls
SPIN_KEYS = {f"spin_{i}{axis}": (i, k) for i in (1, 2) for k, axis in enumerate("xyz")}
POLAR_SPIN_KEYS = ("a_1", "tilt_1", "a_2", "tilt_2", "phi_12", "phi_jl")
#: the global search's coordinates; the amplitude and phase are maximized over
SEARCH_KEYS = ("chirp_mass", "mass_ratio", "theta_jn", "psi", "ra", "dec", TIME_SHIFT)


@dataclasses.dataclass
class Settings:
    """The search's settings (see the module's description)."""
    search_f_min: float = 20.0
    search_bins: int = 256
    search_maxiter: int = 300
    n_searches: int = 3
    continuation: tuple = (20.0, 10.0, 5.0, 3.0)
    mode_separation: float = np.radians(5)
    bins: int = 3000
    tolerance: float = 1e-3
    max_refinements: int = 5
    time_window: float = 3e-3
    n_proc: int = 1
    seed: int = 0
    #: whiten the local searches by the Hessian at their start (else only
    #: by each coordinate's curvature: cheaper, for tests)
    hessian: bool = True
    #: iterations of each local search (None: until converged)
    local_maxiter: int = None


def emulated_trigger(parameters, interferometers, waveform_generator, rng, chirp_mass_error,
                     time_error, **response):
    """What a search reports for the signal ``parameters``: the
    detector-frame chirp mass (relative error ``chirp_mass_error``), a mass
    ratio (error 0.1), the arrival time at the first interferometer (error
    ``time_error``) and the network SNR (error 1)."""
    ifo = interferometers[0]
    arrival = (parameters[TIME_ORIGIN] + parameters[TIME_OFFSET]
               + ifo.time_delay_from_geocenter(parameters["ra"], parameters["dec"],
                                               parameters[TIME_ORIGIN]))
    snr = np.sqrt(_optimal_snr_squared(parameters, interferometers, waveform_generator,
                                       response))
    return dict(
        chirp_mass=parameters["chirp_mass"] * (1 + chirp_mass_error * rng.standard_normal()),
        mass_ratio=float(np.clip(parameters["mass_ratio"] + 0.1 * rng.standard_normal(),
                                 0.5, 1)),
        time=arrival + time_error * rng.standard_normal(),
        snr=snr + rng.standard_normal())


def _optimal_snr_squared(parameters, interferometers, waveform_generator, response,
                         n_grid=20_001):
    """<h|h> on a geometric grid (no data): smooth in f, so the grid need
    not resolve the waveform's oscillations."""
    ifo = interferometers[0]
    freqs = np.geomspace(ifo.minimum_frequency, ifo.maximum_frequency, n_grid)
    return GridModel(Projector(interferometers, waveform_generator, **response),
                     dict(parameters, fiducial=0), freqs).norm_d


class Coordinates:
    """The sampled parameters, with the time as the arrival time at the first
    interferometer, ``dt`` after the trigger's (geocent_time_offset from it,
    the time origin, follows the sky): unlike geocent_time it is nearly
    independent of the sky; and the precessing spins Cartesian
    (``SPIN_KEYS``). Periodic angles are wrapped, the others clipped to
    their prior range (the spin magnitudes radially)."""

    def __init__(self, priors, ifo, trigger_time, time_window):
        if TIME_OFFSET not in priors:
            raise ValueError(f"the search needs the time as {TIME_OFFSET}")
        self.priors = priors
        self.ifo = ifo
        self.trigger_time = trigger_time
        self.time_window = time_window
        keys = [TIME_SHIFT if key == TIME_OFFSET else key
                for key, prior in priors.items()
                if isinstance(prior, Prior) and not isinstance(prior, (DeltaFunction,
                                                                       Constraint))]
        self.precessing = "a_1" in keys
        self.keys = [key for key in keys if key not in POLAR_SPIN_KEYS]
        if self.precessing:
            self.keys += list(SPIN_KEYS)

    def delay(self, parameters):
        return self.ifo.time_delay_from_geocenter(parameters["ra"], parameters["dec"],
                                                  self.trigger_time)

    def bounds(self, keys):
        def bound(key):
            if key == TIME_SHIFT:
                return -self.time_window, self.time_window
            if key in SPIN_KEYS:
                a_max = self.priors[f"a_{SPIN_KEYS[key][0]}"].maximum
                return -a_max, a_max
            return self.priors[key].minimum, self.priors[key].maximum
        return [bound(key) for key in keys]

    @staticmethod
    def spins(parameters):
        """The Cartesian spin vectors of ``SPIN_KEYS``."""
        azimuths = (parameters["phi_jl"], parameters["phi_jl"] + parameters["phi_12"])
        return [parameters[f"a_{i}"] * np.array([np.sin(tilt) * np.cos(phi),
                                                 np.sin(tilt) * np.sin(phi), np.cos(tilt)])
                for i, tilt, phi in zip((1, 2), (parameters["tilt_1"], parameters["tilt_2"]),
                                        azimuths)]

    def polar_spins(self, spins, base):
        """The sampled spin parameters of the vectors ``spins``, the
        magnitudes clipped to their prior; the angles of ``base`` where
        undefined."""
        out = {}
        azimuths = []
        for i, v in zip((1, 2), spins):
            a = np.linalg.norm(v)
            out[f"a_{i}"] = min(a, self.priors[f"a_{i}"].maximum)
            out[f"tilt_{i}"] = np.arccos(np.clip(v[2] / a, -1, 1)) if a > 0 else base[f"tilt_{i}"]
            azimuths.append(np.arctan2(v[1], v[0]) if np.hypot(v[0], v[1]) > 0 else None)
        out["phi_jl"] = base["phi_jl"] if azimuths[0] is None else azimuths[0] % (2 * np.pi)
        out["phi_12"] = (base["phi_12"] if azimuths[1] is None
                         else (azimuths[1] - out["phi_jl"]) % (2 * np.pi))
        return out

    def values(self, parameters, keys):
        spins = self.spins(parameters) if self.precessing else None
        return np.array([parameters[TIME_OFFSET] + self.delay(parameters) if key == TIME_SHIFT
                         else spins[SPIN_KEYS[key][0] - 1][SPIN_KEYS[key][1]]
                         if key in SPIN_KEYS else parameters[key] for key in keys])

    def parameters(self, x, keys, base):
        """``base`` with ``keys`` set to ``x``; the arrival time kept if not
        among them."""
        p = dict(base)
        dt = self.values(base, [TIME_SHIFT])[0]
        spins = self.spins(base) if self.precessing else None
        for key, value, (lo, hi) in zip(keys, x, self.bounds(keys)):
            if key == TIME_SHIFT:
                dt = value
            elif key in SPIN_KEYS:
                i, k = SPIN_KEYS[key]
                spins[i - 1][k] = value
            elif key in PERIODIC:
                p[key] = lo + (value - lo) % (hi - lo)
            else:
                p[key] = min(max(value, lo), hi)
        if any(key in SPIN_KEYS for key in keys):
            p.update(self.polar_spins(spins, base))
        p[TIME_OFFSET] = dt - self.delay(p)
        return p


class Stage:
    """The relative-binning likelihood around ``fiducial``, re-binned from
    the fine summary data ``fine`` (no pass over the data), with
    ``n_bins`` bins geometric over [f_min, the fiducial's maximum
    frequency]."""

    def __init__(self, fine, fiducial, f_min, n_bins):
        self.fiducial = fiducial
        self.likelihood = rebin(fine, fiducial, minimum_frequency=f_min, number_of_bins=n_bins)
        self.evaluations = 0

    def log_likelihood(self, parameters):
        """ln L; -inf where the waveform fails (some points within the
        surrogate's range, not reproduced) or ln L is not finite."""
        self.evaluations += 1
        try:
            value = _log_likelihood_ratio_at(self.likelihood, parameters)
        except ValueError as error:
            logger.warning("%s at %s", error, plain(parameters))
            return -np.inf
        return value if np.isfinite(value) else -np.inf

    def overlaps(self, parameters):
        """<d|h> and <h|h>; (0, 1) where the waveform fails or they are not
        finite."""
        self.evaluations += 1
        try:
            d_h, h_h = overlaps(self.likelihood, parameters)
        except ValueError as error:
            logger.warning("%s at %s", error, plain(parameters))
            return 0j, 1.0
        return (d_h, h_h) if np.isfinite(d_h) and np.isfinite(h_h) and h_h > 0 else (0j, 1.0)

    def maximized_log_likelihood(self, parameters):
        """ln L maximized over the amplitude and an overall phase of the
        waveform, |<d|h>|^2 / (2 ||h||^2): the search statistic, which unlike
        ln L does not flatten out toward large distances away from the peak."""
        dh, hh = self.overlaps(parameters)
        return abs(dh) ** 2 / (2 * hh)


def global_search(stage, coordinates, keys, base, seed, maxiter, popsize=15):
    """Differential evolution of the search statistic over ``keys`` (the rest
    as in ``base``), then the phase and distance which maximize ln L there."""
    bounds = coordinates.bounds(keys)
    x0 = np.clip(coordinates.values(base, keys), *np.array(bounds).T)
    result = differential_evolution(
        lambda x: -stage.maximized_log_likelihood(coordinates.parameters(x, keys, base)),
        bounds, x0=x0, seed=seed, maxiter=maxiter, popsize=popsize, polish=False, tol=0)
    return phase_and_distance(stage, coordinates, coordinates.parameters(result.x, keys, base))


def phase_and_distance(stage, coordinates, parameters, n_phases=32):
    """The phase maximizing ln L maximized over the amplitude,
    max(Re<d|h>, 0)^2 / (2 ||h||^2) (on a grid, then bounded), and the
    distance of that amplitude."""
    lo, hi = coordinates.bounds(["phase"])[0]

    def amplitude_and_log_likelihood(phase):
        dh, hh = stage.overlaps(dict(parameters, phase=phase))
        return dh.real / hh, max(dh.real, 0) ** 2 / (2 * hh)

    grid = lo + (hi - lo) * np.arange(n_phases) / n_phases
    best = grid[np.argmax([amplitude_and_log_likelihood(phase)[1] for phase in grid])]
    step = (hi - lo) / n_phases
    phase = minimize_scalar(lambda phase: -amplitude_and_log_likelihood(phase)[1],
                            bounds=(best - step, best + step), method="bounded").x
    amplitude, _ = amplitude_and_log_likelihood(phase)
    distance = parameters["luminosity_distance"] / max(amplitude, 1e-12)
    parameters = coordinates.parameters([phase, distance], ["phase", "luminosity_distance"],
                                        parameters)
    return parameters, stage.log_likelihood(parameters)


def curvature_scales(function, x0, widths, iterations=3):
    """Per coordinate, the step over which the (negative log-likelihood)
    ``function`` rises by ~1/2 from its second difference, at most the
    coordinate's prior width."""
    f0 = function(x0)
    scales = 1e-3 * widths
    for _ in range(iterations):
        for i in range(len(x0)):
            step = np.zeros_like(x0)
            step[i] = scales[i]
            second = function(x0 + step) + function(x0 - step) - 2 * f0
            scales[i] = widths[i] if second <= 0 else min(scales[i] / np.sqrt(second), widths[i])
    return scales


def hessian(function, x0, step):
    """Central-difference Hessian, in units of ``step`` along each axis."""
    n = len(x0)
    f0 = function(x0)
    e = np.eye(n) * step
    out = np.empty((n, n))
    for i in range(n):
        out[i, i] = (function(x0 + e[i]) + function(x0 - e[i]) - 2 * f0) / step ** 2
        for j in range(i):
            out[i, j] = out[j, i] = (
                function(x0 + e[i] + e[j]) - function(x0 + e[i] - e[j])
                - function(x0 - e[i] + e[j]) + function(x0 - e[i] - e[j])) / (4 * step ** 2)
    return out


def local_search(stage, coordinates, keys, base, floor=1e-2, gtol=1e-3, maximized=False,
                 hessian_whitening=True, maxiter=None):
    """Maximize ln L (or, if ``maximized``, the search statistic) over
    ``keys`` from ``base``: BFGS in coordinates whitened by the Hessian at
    the start (each axis scaled to a unit change in ln L, flat or rising
    directions to their prior width over 1/sqrt(floor)); Powell's method in
    the same coordinates if BFGS makes no progress, as far from the peak,
    where the quadratic model can be bad enough for its first line search
    to fail, or if it reaches a point where the waveform fails. Without
    ``hessian_whitening`` each axis is only scaled; ``maxiter`` caps the
    iterations."""
    x0 = coordinates.values(base, keys)
    widths = np.diff(coordinates.bounds(keys), axis=1)[:, 0]
    objective = stage.maximized_log_likelihood if maximized else stage.log_likelihood

    def negative(x, strict=False):
        value = -objective(coordinates.parameters(x, keys, base))
        if strict and not np.isfinite(value):
            raise FloatingPointError("the waveform failed")
        return value

    scales = curvature_scales(negative, x0, widths)
    if hessian_whitening:
        w, v = np.linalg.eigh(hessian(lambda y: negative(x0 + scales * y), np.zeros(len(x0)),
                                      0.3))
        transform = scales[:, None] * v / np.sqrt(np.maximum(np.abs(w), floor))
    else:
        transform = np.diag(scales)
    limit = {} if maxiter is None else dict(maxiter=maxiter)
    try:
        result = minimize(lambda z: negative(x0 + transform @ z, strict=True),
                          np.zeros(len(x0)), method="BFGS",
                          options=dict(gtol=gtol, eps=1e-4, **limit))
        failed = result.nit == 0
    except FloatingPointError:
        failed = True
    if failed:
        result = minimize(lambda z: negative(x0 + transform @ z), np.zeros(len(x0)),
                          method="Powell", options=dict(xtol=1e-3, ftol=1e-9, **limit))
    return coordinates.parameters(x0 + transform @ result.x, keys, base), -result.fun


def refine(fine, coordinates, parameters, f_min, n_bins, tolerance, max_iterations,
           **search_kwargs):
    """Local maximization from ``f_min``, re-binned around each maximum,
    until the gain is below ``tolerance``; the last maximum at which the
    waveform could be a fiducial (ln L = -inf if none)."""
    log_l = -np.inf
    for iteration in range(max_iterations):
        try:
            stage = Stage(fine, parameters, f_min, n_bins)
        except ValueError as error:  # the waveform fails at the fiducial
            logger.warning("refinement from %g Hz: %s at %s", f_min, error, plain(parameters))
            return previous if iteration else (parameters, -np.inf)
        start = stage.log_likelihood(parameters)
        previous = parameters, log_l
        parameters, log_l = local_search(stage, coordinates, coordinates.keys, parameters,
                                         **search_kwargs)
        logger.info("refinement %d from %g Hz: ln L %.4f -> %.4f (%d evaluations)",
                    iteration + 1, f_min, start, log_l, stage.evaluations)
        if log_l - start < tolerance:
            break
    return parameters, log_l


def polarization_basis(ra, dec):
    """The unit vectors from which ``psi`` is measured, at ``(ra, dec)``: the
    polarization axis is cos(psi) e_1 - sin(psi) e_2 (bilby's convention)."""
    theta = np.pi / 2 - dec
    return (np.array([np.sin(ra), -np.cos(ra), 0.0]),
            np.array([np.cos(ra) * np.cos(theta), np.sin(ra) * np.cos(theta), -np.sin(theta)]))


def sky_images(parameters, coordinates):
    """The 7 images of ``parameters`` under the symmetries of a planar
    detector's response at one site, at rest: rotations of the source by
    multiples of 90 degrees about the detector's normal (the waveform
    rotated with it, ``psi`` transported), which map each arm-pair response
    (aa - bb) / 2 to plus or minus itself, the sign absorbed by the phase;
    and the reflection n -> -n with theta_jn -> pi - theta_jn. The
    detector-frame chirp mass follows each image's orbital Doppler shift
    (constant over the short signal from the search frequency); the arrival
    time at the first interferometer is kept. Over a long signal the
    Earth's rotation breaks the symmetries, which leaves 8 sky modes, nearly
    degenerate at the search frequency."""
    ifo, time_ = coordinates.ifo, coordinates.trigger_time
    dt = coordinates.values(parameters, [TIME_SHIFT])
    gmst = bilby.gw.utils.greenwich_mean_sidereal_time(time_)
    rotation = np.array([[np.cos(gmst), -np.sin(gmst), 0], [np.sin(gmst), np.cos(gmst), 0],
                         [0, 0, 1]])
    k = rotation @ np.cross(ifo.geometry.x, ifo.geometry.y)
    k /= np.linalg.norm(k)
    n = sky_direction(parameters["ra"], parameters["dec"])
    e_1, e_2 = polarization_basis(parameters["ra"], parameters["dec"])
    axis = np.cos(parameters["psi"]) * e_1 - np.sin(parameters["psi"]) * e_2
    doppler = doppler_factor(parameters["ra"], parameters["dec"], time_)
    images = []
    for quarter in range(4):
        for mirror in (False, True):
            if quarter == 0 and not mirror:
                continue
            alpha = quarter * np.pi / 2

            def rotate(v):
                return (v * np.cos(alpha) + np.cross(k, v) * np.sin(alpha)
                        + k * (k @ v) * (1 - np.cos(alpha)))

            m, image = rotate(axis), rotate(n) * (-1 if mirror else 1)
            ra = float(np.arctan2(image[1], image[0]) % (2 * np.pi))
            dec = float(np.arcsin(np.clip(image[2], -1, 1)))
            f_1, f_2 = polarization_basis(ra, dec)
            images.append(coordinates.parameters(dt, [TIME_SHIFT], dict(
                parameters, ra=ra, dec=dec, psi=float(np.arctan2(-(m @ f_2), m @ f_1) % np.pi),
                theta_jn=np.pi - parameters["theta_jn"] if mirror else parameters["theta_jn"],
                chirp_mass=parameters["chirp_mass"] * doppler / doppler_factor(ra, dec, time_))))
    return images


def initial_fiducial(trigger, interferometers, waveform_generator, coordinates, priors,
                     **response):
    """A waveform from the trigger: its masses and time, no spins, overhead
    of the first interferometer, inclined by ``INITIAL_THETA_JN``, the
    distance giving the trigger's SNR."""
    ifo = interferometers[0]
    gmst = bilby.gw.utils.greenwich_mean_sidereal_time(trigger["time"])
    spins = (dict(a_1=0.0, a_2=0.0, tilt_1=0.0, tilt_2=0.0, phi_12=0.0, phi_jl=0.0)
             if coordinates.precessing else dict(chi_1=0.0, chi_2=0.0))
    parameters = dict(
        chirp_mass=trigger["chirp_mass"], mass_ratio=trigger["mass_ratio"], **spins,
        lambda_1=INITIAL_LAMBDA, lambda_2=INITIAL_LAMBDA, theta_jn=INITIAL_THETA_JN, psi=0.0, phase=0.0,
        ra=float((gmst + ifo.longitude_radians) % (2 * np.pi)), dec=ifo.latitude_radians,
        luminosity_distance=1.0, **{TIME_ORIGIN: trigger["time"], TIME_OFFSET: 0.0},
        fiducial=0)
    for key, prior in priors.items():  # fixed in the priors: as fixed there
        if isinstance(prior, DeltaFunction):
            parameters[key] = prior.peak
    parameters = coordinates.parameters([0.0], [TIME_SHIFT], parameters)
    snr_at_1_mpc = np.sqrt(_optimal_snr_squared(parameters, interferometers,
                                                waveform_generator, response))
    lo, hi = priors["luminosity_distance"].minimum, priors["luminosity_distance"].maximum
    parameters["luminosity_distance"] = float(np.clip(snr_at_1_mpc / trigger["snr"], lo, hi))
    return parameters


def plain(parameters):
    return {key: float(value) for key, value in parameters.items()}


class Checkpoint:
    """The search's results so far, in a JSON file (if ``path`` is given):
    a search that is stopped resumes from its last finished step."""

    def __init__(self, path):
        self.path = path
        self.state = {}
        if path and os.path.exists(path):
            with open(path) as f:
                self.state = json.load(f)
            logger.info("resuming the search from %s", path)

    def get(self, key):
        return self.state.get(key)

    def set(self, key, value):
        self.state[key] = value
        if self.path:
            with open(self.path + ".tmp", "w") as f:
                json.dump(self.state, f, indent=1)
            os.replace(self.path + ".tmp", self.path)


def _modes(log_ls_and_parameters):
    return [dict(log_l=float(log_l), parameters=plain(p)) for log_l, p in log_ls_and_parameters]


def _unmodes(entries):
    return [(entry["log_l"], entry["parameters"]) for entry in entries]


def maximum_likelihood(fine, coordinates, fiducial, settings, checkpoint=None):
    """theta_hat and the candidate modes, from the fine summary data
    ``fine`` around the trigger's fiducial: the results of the searches from
    ``settings.search_f_min`` and their sky images (:func:`sky_images`),
    each locally re-maximized there, those within
    ``settings.mode_separation`` on the sky merged, then each refined from
    the decreasing minimum frequencies ``settings.continuation`` in turn.
    With more cycles in band the likelihood has more local maxima (cycle
    slips, ever closer in chirp mass), and the maximum from each minimum
    frequency lies well within the peak of the next, which a search from the
    lowest frequency alone does not reach; the sky modes separate only at
    the lowest frequencies. Returns [(ln L, parameters)], best first."""
    checkpoint = checkpoint if isinstance(checkpoint, Checkpoint) else Checkpoint(checkpoint)
    rng = np.random.default_rng(settings.seed)
    seeds = [int(s) for s in rng.integers(2 ** 32, size=settings.n_searches)]
    distinct = checkpoint.get("sky modes")
    if distinct is None:
        search = Stage(fine, fiducial, settings.search_f_min, settings.search_bins)
        modes = []
        for k, seed in enumerate(seeds):
            key = f"search {k + 1}"
            found = checkpoint.get(key)
            if found is None:
                start = time.perf_counter()
                found, log_l = global_search(search, coordinates, SEARCH_KEYS, fiducial, seed,
                                             settings.search_maxiter)
                logger.info("search %d from %g Hz: ln L_RB %.2f (%.0f s)", k + 1,
                            settings.search_f_min, log_l, time.perf_counter() - start)
                found = [(log_l, found)]
                for image in sky_images(found[0][1], coordinates):
                    image, _ = local_search(search, coordinates, SEARCH_KEYS, image,
                                            maximized=True, **_search_kwargs(settings))
                    found.append(phase_and_distance(search, coordinates, image)[::-1])
                checkpoint.set(key, _modes(found))
            else:
                found = _unmodes(found)
            modes += found
        distinct = []
        for log_l, parameters in sorted(modes, key=lambda mode: -mode[0]):
            n = sky_direction(parameters["ra"], parameters["dec"])
            if all(np.arccos(np.clip(n @ sky_direction(other["ra"], other["dec"]), -1, 1))
                   > settings.mode_separation for _, other in distinct):
                distinct.append((log_l, parameters))
        logger.info("%d sky modes from %g Hz: ln L_RB %s", len(distinct), settings.search_f_min,
                    ", ".join(f"{log_l:.2f}" for log_l, _ in distinct))
        checkpoint.set("sky modes", _modes(distinct))
    else:
        distinct = _unmodes(distinct)

    candidates = [None] * len(distinct)
    tasks = []
    for k, (_, parameters) in enumerate(distinct):
        done = checkpoint.get(f"mode {k + 1}")
        if done is None:
            tasks.append((k + 1, parameters))
        else:
            candidates[k] = _unmodes([done])[0]
    state = dict(fine=fine, coordinates=coordinates, settings=settings)
    if tasks and settings.n_proc > 1:  # spawned: JAX's threads do not survive a fork
        with multiprocessing.get_context("spawn").Pool(
                min(settings.n_proc, len(tasks)), initializer=_init_modes,
                initargs=(state, logger.getEffectiveLevel())) as pool:
            for k, result in pool.imap_unordered(_continue_mode, tasks):
                candidates[k - 1] = result
                checkpoint.set(f"mode {k}", _modes([result])[0])
    else:
        _MODES.update(state)
        for task in tasks:
            k, result = _continue_mode(task)
            candidates[k - 1] = result
            checkpoint.set(f"mode {k}", _modes([result])[0])
    candidates.sort(key=lambda c: -c[0])
    return candidates


def _search_kwargs(settings):
    return dict(hessian_whitening=settings.hessian, maxiter=settings.local_maxiter)


#: what the workers of :func:`_continue_mode` need
_MODES = {}


def _init_modes(state, level):
    _MODES.update(state)
    logger.setLevel(level)


def _continue_mode(task):
    """Refine the sky mode ``task`` = (number, parameters) from each of the
    minimum frequencies ``settings.continuation`` in turn."""
    k, parameters = task
    settings = _MODES["settings"]
    for f_min in settings.continuation:
        parameters, log_l = refine(_MODES["fine"], _MODES["coordinates"], parameters, f_min,
                                   settings.bins, settings.tolerance, settings.max_refinements,
                                   **_search_kwargs(settings))
        logger.info("mode %d, refined from %g Hz: ln L_RB %.4f", k, f_min, log_l)
        if not np.isfinite(log_l):
            break
    return k, (log_l, parameters)


#: the keyword arguments of the detector response, which the searches'
#: projections take from the likelihood's
RESPONSE_KEYS = ("earth_rotation_time_delay", "earth_rotation_beam_patterns", "finite_size",
                 "orbital_motion")


def maximum_likelihood_point(interferometers, waveform_generator, data, trigger, priors,
                             settings, fine_bins=100_000, outdir=None, **likelihood_kwargs):
    """The search from the data and the trigger alone: the fiducial from the
    trigger (:func:`initial_fiducial`), the one pass over ``data`` with
    ``fine_bins`` bins around it (:func:`bilby_xG.data.fine_likelihood`,
    kept in ``outdir`` as ``fine_likelihood.pickle``), then
    :func:`maximum_likelihood` (checkpointed in ``outdir`` as
    ``search.json``). ``likelihood_kwargs`` go to the likelihood (the
    detector response, ``second_order``, ...). Returns the candidates
    [(ln L, parameters)], best first, and the fine likelihood."""
    import pickle
    from .data import fine_likelihood

    response = {key: likelihood_kwargs[key] for key in RESPONSE_KEYS if key in likelihood_kwargs}
    coordinates = Coordinates(priors, interferometers[0], trigger["time"], settings.time_window)
    fiducial = initial_fiducial(trigger, interferometers, waveform_generator, coordinates,
                                priors, **response)
    logger.info("initial fiducial %s", plain(fiducial))
    path = outdir and os.path.join(outdir, "fine_likelihood.pickle")
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            fine = pickle.load(f)
        logger.info("fine summary data from %s", path)
    else:
        start = time.perf_counter()
        fine = fine_likelihood(interferometers, waveform_generator, data, fiducial, fine_bins,
                               priors=priors, **likelihood_kwargs)
        logger.info("fine summary data, %d bins: %.0f s", fine.number_of_bins,
                    time.perf_counter() - start)
        if path:
            with open(path + ".tmp", "wb") as f:
                pickle.dump(fine, f)
            os.replace(path + ".tmp", path)
    candidates = maximum_likelihood(
        fine, coordinates, fiducial, settings,
        checkpoint=outdir and os.path.join(outdir, "search.json"))
    logger.info("theta_hat %s, ln L_RB %.4f", plain(candidates[0][1]), candidates[0][0])
    return candidates, fine
