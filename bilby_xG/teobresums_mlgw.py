r"""TEOBResumS in mlgw_bns' conventions, as per-mode sources: the data of
injections recovered with mlgw_bns (:func:`bilby_xG.source.mlgw_bns_individual_modes`,
:func:`bilby_xG.source.mlgw_bns_precessing_individual_modes`), with the same
arguments, conventions and output.

Both run TEOBResumS' precessing frequency-domain model (``use_spins = 2``;
aligned spins as zero in-plane spins, as mlgw_bns' precessing validation
does) with the settings the shipped model ``mlgw_bns_model`` was trained on
(``Giotto``, its training ``srate_interp`` in units of the total mass), the
ODE started low enough for the highest-m mode to cover ``minimum_frequency``:

- each mode is one TEOBResumS call (``use_mode_lm = [k]``; the calls sum to
  the all-mode one, to 2e-16): the twisted polarisations of the
  co-precessing mode :math:`(\ell, m')` (for aligned spins, of the
  :math:`(\ell, \pm m)` pair), keyed ``"l,m"``, with its own time-frequency
  map ``"time_to_merger"`` from its co-precessing phase;
- the merger is where the surrogate puts it, the time of the tangent to the
  co-precessing (2,2) phase at the top of the surrogate's trained band
  (TEOBResumS' own, its ``tc``, is ~65 us away for GW170817-like BNSs); or,
  with a ``time_reference_frequency``
  (:func:`bilby_xG.source.mlgw_bns_time_reference`), the time is that of the
  co-precessing (2,2) at that frequency, below the merger, where TEOBResumS'
  time-frequency map is smooth in the parameters to ~0.02 us (above, it
  scatters by up to several us unless its SPA tail is attached between the
  samples);
- TEOBResumS imposes the spins, and zeroes its orbital phase, where its ODE
  starts. The binary is given at ``reference_frequency`` (aligned spins: at
  the merger): the spins, L and the line of sight are carried to the start
  by mlgw_bns' PN precession (:func:`mlgw_bns.batched_precession.integrate_angles`),
  and the binary turned rigidly about L (spins +d, azimuth -d) by the d that
  maximises the overlap with the surrogate at the binary (both polarisations,
  nothing else optimised). That fixes TEOBResumS' orbital phase to the
  binary's to the surrogate's phase error; nothing else is fitted.

TEOBResumS' output phases slip by 2 pi from point to point on grids too
coarse for them (the complex values stay right): every call is on the
requested frequencies merged with :func:`slip_free_grid`, so the phases, and
the times from them, are continuous.

With the 2000 model at a GW170817-like injection (tests/test_teobresums_mlgw.py,
nothing optimised): aligned ~2e-8, precessing ~2e-7, the latter the PN
transport of the spins from 20 Hz to TEOBResumS' start (~1.3 Hz), which its
own spin evolution undoes slightly differently.
"""
import dataclasses

import numpy as np
from scipy.interpolate import CubicSpline

from .source import (MLGW_BNS_MODES, _check_mlgw_bns_modes, _mlgw_bns_model,
                     mlgw_bns_individual_modes, mlgw_bns_model_name,
                     mlgw_bns_time_reference)

__author__ = ["Jacopo Tissino"]

try:
    from mlgw_bns.constants import SUN_MASS_SECONDS as SUN_SECONDS, TEOBRESUMS_MASS_FACTOR
except ImportError:  # mlgw_bns before its constants module: the same values
    #: G Msun / c^3 (s), IAU 2015 nominal (= LAL MTSUN_SI, bilby)
    SUN_SECONDS = 4.925490947641267e-06
    #: a mass to TEOBResumS in physical units, times this: its MSUN_S
    #: (4.925491025543576e-6, TEOBResumS.h) is 1.6e-8 above the IAU value
    TEOBRESUMS_MASS_FACTOR = SUN_SECONDS / 4.925491025543575903411922162094833998e-6
#: TEOBResumS' frequency-domain evolution starts this fraction below its band
#: (SPA_F0_SAFETY in TEOBResumS.h), where it imposes the spins
TEOBRESUMS_SPA_F0_SAFETY = 0.05
#: the ODE starts this factor below the frequency where the highest-m mode
#: reaches ``minimum_frequency``
START_SAFETY = 1.1
#: aligned spins: the largest relative squared difference, at a test turn,
#: between the joint call and the single-mode calls times their phases
#: before the fit falls back to calling TEOBResumS at every turn
ALIGNED_TOLERANCE = 1e-8
#: grid of the fit of the turn about L (Hz)
FIT_BAND = (25.0, 2000.0)
FIT_POINTS = 4000
#: largest advance (rad) of the fastest mode's phase between the points TEOBResumS
#: is evaluated at, below pi
SLIP_FREE_STEP = 2.5
#: points of the slip-free grid beyond the requested frequencies on each side
#: of the local splines of :meth:`TEOBResumSRun.modes` (their end conditions'
#: influence decays by ~0.27 a point)
DENSE_MARGIN = 16


def slip_free_grid(total_mass, mass_ratio, minimum_frequency, maximum_frequency, m_max=4):
    """Frequencies (Hz) on which every mode up to ``m_max``'s phase advances
    by at most :data:`SLIP_FREE_STEP` from point to point (by the 0PN time to
    merger, which overestimates it), plus a 1e-3 geometric grid."""
    eta = mass_ratio / (1 + mass_ratio) ** 2
    chirp_mass_s = total_mass * eta ** 0.6 * SUN_SECONDS
    # (2,2): t(f) = A f^(-8/3); mode m at f: t(2 f / m); phase rate 2 pi t
    A = 5 / 256 * chirp_mass_s ** (-5 / 3) * np.pi ** (-8 / 3)
    # the fastest mode's phase is c f^(-5/3): uniform steps in f^(-5/3)
    c = 2 * np.pi * A * (m_max / 2) ** (8 / 3) * 3 / 5
    u_lo, u_hi = minimum_frequency ** (-5 / 3), maximum_frequency ** (-5 / 3)
    n = int(np.ceil(c * (u_lo - u_hi) / SLIP_FREE_STEP)) + 1
    chirp = np.linspace(u_lo, u_hi, n) ** (-3 / 5)
    geometric = np.geomspace(minimum_frequency, maximum_frequency,
                             int(np.log(maximum_frequency / minimum_frequency) / 1e-3) + 1)
    return np.union1d(chirp, geometric)


def _chirp_coordinate(frequencies):
    """x = -f^(-5/3), increasing with f: the inspiral phase is nearly linear in it"""
    return -np.asarray(frequencies, dtype=float) ** (-5 / 3)


def _chirp_coordinate_derivative(frequencies):
    """dx / df"""
    return 5 / 3 * np.asarray(frequencies, dtype=float) ** (-8 / 3)


#: relative half-width and points of the window in which the time at a
#: ``time_reference_frequency`` is fitted
TIME_REFERENCE_WINDOW = 2e-3
TIME_REFERENCE_POINTS = 513


def _merger_top(model, total_mass):
    """The two points (Hz) of the surrogate's merger reference, the top of
    the (2,2)'s trained band (as Model.teob_modes_amp_phase)."""
    from mlgw_bns.higher_order_modes import Mode

    reference = model.mode_models[Mode(2, 2)]
    f_top = float(reference.dataset.frequencies[reference.downsampling_indices.phase_indices[-1]])
    return np.array([f_top * (1 - 1e-3), f_top]) / (total_mass * SUN_SECONDS)


def _band(freqs, minimum_frequency):
    if minimum_frequency is None:
        positive = freqs[freqs > 0]
        minimum_frequency = float(positive[0]) if len(positive) else 1.0
    return float(minimum_frequency), freqs >= minimum_frequency


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _ry(b):
    c, s = np.cos(b), np.sin(b)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def transported(P, frequency):
    """``P`` (mlgw_bns' PrecessingParametersWithExtrinsic, the spins at its
    reference frequency) with the spins, L and the line of sight carried to
    ``frequency`` by mlgw_bns' PN precession, in the frame z = L there."""
    from mlgw_bns.batched_precession import integrate_angles
    from mlgw_bns.twist_waveform import nu_to_X1

    q, M = P.mass_ratio, P.total_mass
    _, state, _ = integrate_angles(q, P.lambda_1, P.lambda_2, np.asarray(P.chi_1),
                                   np.asarray(P.chi_2), P.reference_frequency_hz * M * SUN_SECONDS,
                                   frequency * M * SUN_SECONDS, full_state=True)
    s0 = np.asarray(state)[0]
    X1 = nu_to_X1(q / (1 + q) ** 2)
    L = s0[6:9]
    R = _rz(np.arctan2(L[1], L[0])) @ _ry(np.arccos(np.clip(L[2] / np.linalg.norm(L), -1, 1)))
    chi1, chi2 = R.T @ (s0[0:3] / X1 ** 2), R.T @ (s0[3:6] / (1 - X1) ** 2)
    # the azimuth enters TEOBResumS with the opposite sign (xg_pe's ps_teob_probe)
    N = np.array([np.sin(P.inclination) * np.cos(-P.azimuth),
                  np.sin(P.inclination) * np.sin(-P.azimuth), np.cos(P.inclination)])
    Np = R.T @ N
    return dataclasses.replace(P, chi_1=tuple(chi1), chi_2=tuple(chi2),
                               inclination=float(np.arccos(np.clip(Np[2], -1, 1))),
                               azimuth=float(-np.arctan2(Np[1], Np[0])),
                               reference_frequency_hz=float(frequency))


def rotated(P, delta):
    """the same binary rigidly turned about L: spins +delta, azimuth -delta"""
    Rd = _rz(delta)
    return dataclasses.replace(P, chi_1=tuple(Rd @ np.asarray(P.chi_1)),
                               chi_2=tuple(Rd @ np.asarray(P.chi_2)), azimuth=P.azimuth - delta)


class TEOBResumSRun:
    """TEOBResumS for one binary: ``start`` (mlgw_bns'
    PrecessingParametersWithExtrinsic, the spins at TEOBResumS' spin
    frequency ``0.95 f0``, L along z) turned about L by the ``delta`` that
    maximises the overlap with ``target(frequencies) -> (hp, hc)`` (mlgw_bns'
    convention), with the surrogate's merger reference."""

    def __init__(self, model, start, f0, minimum_frequency, target,
                 time_reference=(None, 0.0)):
        self.model, self.start, self.f0 = model, start, f0
        self.time_reference = time_reference
        m_max = max(mode.m for mode in model.modes)
        self.top = _merger_top(model, start.total_mass)
        self.slip_free = slip_free_grid(start.total_mass, start.mass_ratio, minimum_frequency,
                                        1.05 * self.top[1], m_max)
        self.t_merger = None
        self._dense = {}
        self.delta = self._fit_delta(target)

    @classmethod
    def frequencies(cls, model, minimum_frequency):
        """(f0, the spin frequency) of TEOBResumS' start for ``model``'s modes"""
        m_max = max(mode.m for mode in model.modes)
        f0 = 2 * minimum_frequency / m_max / START_SAFETY
        return f0, (1 - TEOBRESUMS_SPA_F0_SAFETY) * f0

    def run(self, frequencies, k, delta=None):
        """One TEOBResumS call for the multipoles ``k`` (TEOBResumS indices)
        at ``frequencies`` (increasing, above f0): (hp, hc, hflm, dynamics),
        the polarisations in mlgw_bns' Fourier convention, TEOBResumS' time."""
        from mlgw_bns.dataset_generation import eobrun
        from mlgw_bns.higher_order_modes import Mode, mode_to_k, srate_interp_scaling

        P = rotated(self.start, self.delta if delta is None else delta)
        largest_l = max(mode.l for mode in self.model.modes)
        dataset = self.model.dataset
        _, hp_re, hp_im, hc_re, hc_im, hflm, _, dyn = eobrun(dict(
            # M: TEOBResumS' MSUN_S -> IAU
            M=P.total_mass * TEOBRESUMS_MASS_FACTOR, q=P.mass_ratio, LambdaAl2=P.lambda_1, LambdaBl2=P.lambda_2,
            chi1x=P.chi_1[0], chi1y=P.chi_1[1], chi1z=P.chi_1[2],
            chi2x=P.chi_2[0], chi2y=P.chi_2[1], chi2z=P.chi_2[2],
            distance=P.distance_mpc, inclination=P.inclination,
            coalescence_angle=np.pi / 2 + P.azimuth, use_spins=2, domain=1,
            initial_frequency=self.f0,
            srate_interp=dataset.srate_hz * dataset.total_mass / P.total_mass
            * srate_interp_scaling(self.model.modes),
            interp_freqs="yes", freqs=[self.f0, *frequencies], use_geometric_units="no",
            output_hpc="no", arg_out="yes", use_mode_lm=list(k),
            use_mode_lm_inertial=list(range(mode_to_k(Mode(largest_l, largest_l)) + 1))))
        hp = np.conj(np.asarray(hp_re) + 1j * np.asarray(hp_im))[1:]
        hc = np.conj(np.asarray(hc_re) + 1j * np.asarray(hc_im))[1:]
        return hp, hc, hflm, dyn

    def merger_time(self, delta=None):
        """The polarisations' merger time at the surrogate's reference: the
        tangent to the co-precessing (2,2) phase at the top of its trained
        band (with a time reference frequency, what puts the tangent there at
        minus its offset). hflm's phase (minus mlgw_bns') carries -2 pi f tc, tc
        (dyn["tc"]) in units of M, which the polarisations do not. Kept for
        the fitted turn."""
        if delta is not None or self.t_merger is None:
            from mlgw_bns.higher_order_modes import Mode, mode_to_k
            k22 = mode_to_k(Mode(2, 2))
            f_time, offset = self.time_reference
            if f_time is not None:
                # the slope of the phase there; points close enough for
                # TEOBResumS' phase not to slip
                x = np.linspace(-1, 1, TIME_REFERENCE_POINTS)
                _, _, hflm, dyn = self.run(f_time * (1 + TIME_REFERENCE_WINDOW * x), [k22], delta)
                phase = np.unwrap(-np.asarray(hflm[str(k22)][1])[1:])
                slope = np.polynomial.Polynomial.fit(
                    x, phase - phase[len(x) // 2], 4, domain=[-1, 1]).deriv()(0.0)
                t_reference = (slope / (TIME_REFERENCE_WINDOW * f_time) / (2 * np.pi)
                               + float(dyn["tc"]) - offset)
                if delta is None:
                    self.t_merger = t_reference
                return t_reference
            # TEOBResumS' interpolation needs more than the two points
            grid = np.union1d(np.geomspace(self.top[1] / 2, self.top[1], 16), self.top)
            at = np.searchsorted(grid, self.top)
            _, _, hflm, dyn = self.run(grid, [k22], delta)
            phase = -np.asarray(hflm[str(k22)][1])[1:][at]
            t_merger = (float(phase[1] - phase[0]) / float(self.top[1] - self.top[0])
                        / (2 * np.pi) + float(dyn["tc"]))
            if delta is not None:
                return t_merger
            self.t_merger = t_merger
        return self.t_merger

    def polarisations(self, frequencies, delta=None):
        """(hp, hc) summed over the model's modes, referenced to the merger."""
        from mlgw_bns.higher_order_modes import mode_to_k
        k = sorted(mode_to_k(mode) for mode in self.model.modes)
        hp, hc, _, _ = self.run(frequencies, k, delta)
        shift = np.exp(-2j * np.pi * self.merger_time(delta) * np.asarray(frequencies))
        return hp * shift, hc * shift

    def modes(self, frequencies, modes):
        """{(l, m): (hp, hc, time_to_merger)} at ``frequencies`` (increasing,
        above f0): interpolated from each mode's one call on the slip-free
        grid (:meth:`_dense_mode`), or, outside that grid, each mode's own
        call on them merged with it (:meth:`modes_direct`)."""
        frequencies = np.asarray(frequencies, dtype=float)
        if not len(frequencies) or frequencies[0] < self.slip_free[0] \
                or frequencies[-1] > self.slip_free[-1]:
            return self.modes_direct(frequencies, modes)
        shift = np.exp(-2j * np.pi * self.merger_time() * frequencies)
        x = _chirp_coordinate(frequencies)
        out = {}
        for ell, emm in modes:
            grid_x, phase, plus, cross = self._dense_mode(ell, emm)
            lo = max(int(np.searchsorted(grid_x, x[0])) - DENSE_MARGIN, 0)
            hi = min(int(np.searchsorted(grid_x, x[-1])) + DENSE_MARGIN, len(grid_x))
            knots = grid_x[lo:hi]
            spline = CubicSpline(knots, phase[lo:hi])
            co_phase = spline(x)
            carrier = np.exp(1j * co_phase)
            hp = CubicSpline(knots, plus[lo:hi])(x) * carrier
            hc = CubicSpline(knots, cross[lo:hi])(x) * carrier
            # the co-precessing mode's own time-frequency map, about the merger
            time = spline(x, 1) * _chirp_coordinate_derivative(frequencies) / (2 * np.pi) \
                - self.merger_time()
            out[(ell, emm)] = (hp * shift, hc * shift, time)
        return out

    def _dense_mode(self, ell, emm):
        """Mode (l, m)'s one TEOBResumS call on the slip-free grid, kept as
        functions of x = -f^(-5/3) (in which the inspiral phase is nearly
        linear) smooth enough to interpolate: (x, its co-precessing phase
        psi, hp and hc demodulated by it). psi is hflm's phase without its
        -2 pi f tc (which the polarisations do not carry), unwrapped (the
        slip-free grid's steps are below pi), so psi' / 2 pi is the time."""
        if (ell, emm) not in self._dense:
            from mlgw_bns.higher_order_modes import Mode, mode_to_k
            k = mode_to_k(Mode(ell, emm))
            hp, hc, hflm, dyn = self.run(self.slip_free, [k])
            psi = np.unwrap(-np.asarray(hflm[str(k)][1])[1:]
                            + 2 * np.pi * self.slip_free * float(dyn["tc"]))
            demodulate = np.exp(-1j * psi)
            self._dense[(ell, emm)] = (_chirp_coordinate(self.slip_free), psi,
                                       hp * demodulate, hc * demodulate)
        return self._dense[(ell, emm)]

    def modes_direct(self, frequencies, modes):
        """{(l, m): (hp, hc, time_to_merger)} at ``frequencies`` (increasing,
        above f0), each mode its own call on them merged with the slip-free
        grid."""
        from mlgw_bns.higher_order_modes import Mode, mode_to_k
        grid = np.union1d(frequencies, self.slip_free)
        at = np.searchsorted(grid, frequencies)
        shift = np.exp(-2j * np.pi * self.merger_time() * frequencies)
        out = {}
        for ell, emm in modes:
            k = mode_to_k(Mode(ell, emm))
            hp, hc, hflm, dyn = self.run(grid, [k])
            # the co-precessing mode's own time-frequency map, about the merger
            co_phase = -np.asarray(hflm[str(k)][1])[1:]
            time = (np.gradient(co_phase, grid) / (2 * np.pi))[at] + float(dyn["tc"]) \
                - self.merger_time()
            out[(ell, emm)] = (hp[at] * shift, hc[at] * shift, time)
        return out

    def _aligned_polarisations(self, grid):
        """For spins along L, ``delta -> (hp, hc)`` at ``delta`` without new
        TEOBResumS calls, or None: turning about L then only moves the
        azimuth, so each mode is its own call at 0 times exp(i s m delta),
        and the merger time does not change. The sign s, and that the
        single-mode calls sum to the joint one, from one direct call."""
        if np.any(np.asarray(self.start.chi_1)[:2]) or np.any(np.asarray(self.start.chi_2)[:2]):
            return None
        from mlgw_bns.higher_order_modes import mode_to_k
        shift = np.exp(-2j * np.pi * self.merger_time(0.0) * grid)
        parts = []
        for mode in self.model.modes:
            hp, hc, _, _ = self.run(grid, [mode_to_k(mode)], 0.0)
            parts.append((mode.m, hp * shift, hc * shift))

        def polarisations(delta, sign):
            turn = [np.exp(1j * sign * m * delta) for m, _, _ in parts]
            return (sum(t * hp for t, (_, hp, _) in zip(turn, parts)),
                    sum(t * hc for t, (_, _, hc) in zip(turn, parts)))
        probe = 0.5
        hp_d, hc_d = self.polarisations(grid, probe)
        scale = np.sum(abs(hp_d) ** 2 + abs(hc_d) ** 2)
        miss = {sign: np.sum(abs(polarisations(probe, sign)[0] - hp_d) ** 2
                             + abs(polarisations(probe, sign)[1] - hc_d) ** 2) / scale for sign in (1, -1)}
        sign = min(miss, key=miss.get)
        if miss[sign] > ALIGNED_TOLERANCE:
            return None
        return lambda delta: polarisations(delta, sign)

    def _fit_delta(self, target):
        grid = np.geomspace(*FIT_BAND, FIT_POINTS)
        weight = np.gradient(grid) / grid ** (7 / 3)
        hp_m, hc_m = target(grid)
        norm_m = np.sum((abs(hp_m) ** 2 + abs(hc_m) ** 2) * weight)
        # aligned spins: the turn is a phase per mode (9 calls instead of ~90)
        fast = self._aligned_polarisations(grid)
        polarisations = fast if fast is not None else (lambda delta: self.polarisations(grid, delta))

        def overlap(delta):
            hp, hc = polarisations(delta)
            norm = np.sqrt(np.sum((abs(hp) ** 2 + abs(hc) ** 2) * weight) * norm_m)
            return float(np.sum((np.conj(hp) * hp_m + np.conj(hc) * hc_m) * weight).real / norm)

        deltas = np.linspace(-np.pi, np.pi, 36, endpoint=False)
        best = float(deltas[int(np.argmax([overlap(d) for d in deltas]))])
        step = float(deltas[1] - deltas[0])
        for _ in range(3):  # parabolic refinements
            y0, y1, y2 = (overlap(best + x) for x in (-step, 0.0, step))
            best += 0.5 * step * (y0 - y2) / (y0 - 2 * y1 + y2)
            step /= 6
        self.overlap = overlap(best)
        return best


_RUNS = {}
_RUNS_SIZE = 4
_LAST = {}


def _run(binary, make):
    if binary not in _RUNS:
        if len(_RUNS) >= _RUNS_SIZE:
            _RUNS.pop(next(iter(_RUNS)))
        _RUNS[binary] = make()
    return _RUNS[binary]


def _source(run, binary, freqs, in_band, modes, cross_sign):
    """the per-mode output on ``freqs`` (zero outside ``in_band``); the same
    binary at the same frequencies (per interferometer) reused"""
    key = (binary, tuple(modes), len(freqs), hash(freqs.tobytes()))
    if _LAST.get("key") != key:
        def full(values, dtype=complex):
            x = np.zeros(len(freqs), dtype=dtype)
            x[in_band] = values
            return x

        out = {f"{ell},{emm}": {"plus": full(hp), "cross": full(cross_sign * hc),
                                "time_to_merger": full(time, float)}
               for (ell, emm), (hp, hc, time) in run.modes(freqs[in_band], modes).items()}
        _LAST.update(key=key, out=out)
    return _LAST["out"]


def teobresums_mlgw_individual_modes(frequency_array, M, q, chi1z, chi2z, LambdaAl2,
                                     LambdaBl2, distance, inclination, coalescence_angle,
                                     **kwargs):
    """Aligned-spin tidal TEOBResumS, as :func:`bilby_xG.source.mlgw_bns_individual_modes`
    (same arguments, conventions, ``hx`` sign and output), zero below
    ``minimum_frequency``. Use with
    :func:`bilby_xG.source.convert_to_mlgw_bns_parameters`."""
    from mlgw_bns.precessing_model import PrecessingParametersWithExtrinsic

    freqs = np.asarray(kwargs.get("frequency_bin_edges", frequency_array), dtype=float)
    modes = _check_mlgw_bns_modes(kwargs.get("mode_array", MLGW_BNS_MODES))
    f_min, in_band = _band(freqs, kwargs.get("minimum_frequency"))
    name = mlgw_bns_model_name(kwargs.get("mlgw_bns_model"))
    q = max(q, 1.0 / q)  # m1 / m2 >= 1
    time_reference = mlgw_bns_time_reference(kwargs)
    binary = ("aligned", M, q, chi1z, chi2z, LambdaAl2, LambdaBl2, distance, inclination,
              coalescence_angle, f_min, name, time_reference)

    def make():
        model = _mlgw_bns_model(name)
        f0, f_spins = TEOBResumSRun.frequencies(model, f_min)
        start = PrecessingParametersWithExtrinsic(
            q, LambdaAl2, LambdaBl2, (0.0, 0.0, chi1z), (0.0, 0.0, chi2z), distance,
            inclination, M, reference_frequency_hz=f_spins)

        def target(grid):
            surrogate = mlgw_bns_individual_modes(
                grid, M, q, chi1z, chi2z, LambdaAl2, LambdaBl2, distance, inclination,
                coalescence_angle, mlgw_bns_model=name, mode_array=MLGW_BNS_MODES,
                time_reference_frequency=time_reference[0],
                time_reference_offset=time_reference[1])
            return (sum(m["plus"] for m in surrogate.values()),
                    sum(m["cross"] for m in surrogate.values()))

        return TEOBResumSRun(model, start, f0, f_min, target, time_reference)

    return _source(_run(binary, make), binary, freqs, in_band, modes, 1)


teobresums_mlgw_individual_modes.evaluates_on_bin_edges = True


def teobresums_mlgw_precessing_individual_modes(frequency_array, M, q, a_1, a_2, tilt_1,
                                                tilt_2, phi_12, phi_jl, LambdaAl2, LambdaBl2,
                                                distance, theta_jn, phase, **kwargs):
    r"""Precessing tidal TEOBResumS, as
    :func:`bilby_xG.source.mlgw_bns_precessing_individual_modes` (same
    arguments, LVK spins at ``reference_frequency``, conventions, ``hx`` with
    LALSuite's sign), zero below ``minimum_frequency``; the first call for a
    binary fits its turn about L (~1 min). Use with
    :func:`bilby_xG.source.convert_to_mlgw_bns_precessing_parameters`."""
    from mlgw_bns.precessing_model import PrecessingModel, PrecessingParametersWithExtrinsic

    freqs = np.asarray(kwargs.get("frequency_bin_edges", frequency_array), dtype=float)
    modes = _check_mlgw_bns_modes(kwargs.get("mode_array", MLGW_BNS_MODES))
    f_min, in_band = _band(freqs, kwargs.get("minimum_frequency"))
    reference_frequency = float(kwargs.get("reference_frequency", 50.0))
    name = mlgw_bns_model_name(kwargs.get("mlgw_bns_model"))
    q = max(q, 1.0 / q)  # m1 / m2 >= 1
    time_reference = mlgw_bns_time_reference(kwargs)
    binary = ("precessing", M, q, a_1, a_2, tilt_1, tilt_2, phi_12, phi_jl, LambdaAl2,
              LambdaBl2, distance, theta_jn, phase, reference_frequency, f_min, name,
              time_reference)

    def make():
        model = _mlgw_bns_model(name)
        f0, f_spins = TEOBResumSRun.frequencies(model, f_min)
        P = PrecessingParametersWithExtrinsic.from_lvk(
            M * q / (1 + q), M / (1 + q), theta_jn, phi_jl, tilt_1, tilt_2, phi_12, a_1, a_2,
            phase, reference_frequency, LambdaAl2, LambdaBl2, distance)
        if time_reference[0] is not None:
            P = dataclasses.replace(P, time_reference_frequency_hz=time_reference[0],
                                    merger_time=-time_reference[1])
        precessing = PrecessingModel(model)
        return TEOBResumSRun(model, transported(P, f_spins), f0, f_min,
                             lambda grid: precessing.predict(grid, P), time_reference)

    # hx with LALSuite's sign, as mlgw_bns_precessing_individual_modes
    return _source(_run(binary, make), binary, freqs, in_band, modes, -1)


teobresums_mlgw_precessing_individual_modes.evaluates_on_bin_edges = True
