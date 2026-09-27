"""How far from uniform is the motion of the detectors over a GW170817-like
signal, and does it matter?

The arrival time at a detector follows its barycentric position
``R(t) = r_earth(t) + Rot(t) vertex`` (see :mod:`bilby_xG.motion`). Over
isotropic sky positions and coalescence times uniform over a year, for a
GW170817-like binary neutron star, this computes

* the displacement of the detectors from a uniform motion (their velocity
  at coalescence), ``R(t) - R(t_c) - V(t_c) (t - t_c)``, in 3D and along the
  line of sight, split into the orbit of the geocentre (newly modelled) and
  the rotation of the vertex (already modelled by bilby_xG);
* the dephasing, ``2 pi f n.[R(t_f) - R(t_c) - V(t_c)(t_f - t_c)]/c``, as a
  function of frequency and of time to coalescence, before and after the
  optimal Doppler shift, time and phase;
* the network mismatch between the signal and the signal with the same
  motion made uniform, optimally Doppler shifted, for the orbit alone
  (``orbital_motion=True`` against its uniform approximation, with the
  rotation in both) and for the whole motion of the detectors.

To first order in v/c a common uniform motion only multiplies all the
detector-frame masses (and the distance) by a Doppler factor, which to this
order is the same as adding ``beta * tau(f)`` to the arrival time, with
``tau`` the time to coalescence; the differences between the velocities of
the detectors are known once the sky position is. The uniform-motion
templates are therefore the signal with every detector moving at its
velocity at coalescence, times ``exp(-2 pi i f (beta tau(f) + dt) + i phi)``
with ``beta``, ``dt`` and ``phi`` common to the network. The mismatch is
maximised over ``phi`` analytically and over ``(beta, dt)`` numerically
(from the weighted-least-squares solution for the residual phase), with
``beta`` either zero (the Doppler shift at coalescence) or free. A third,
linearised variant also lets the chirp mass and mass ratio absorb what they
can (the Newtonian ``f^(-5/3)`` and 1PN ``f^(-1)`` phase terms). The
templates share the beam patterns of the signal, so the overlap is a sum
over detectors and frequencies of ``4 |h_ifo|^2 / S_ifo df`` weights times
the phase of the residual. The reference point (center) of the arrival time
does not matter: it only shifts the time.

Usage: python orbital_motion_curvature.py [outdir] [n_samples]
"""
import json
import sys
from pathlib import Path

import bilby
import matplotlib.pyplot as plt
import numpy as np
from bilby.core.utils import speed_of_light
from scipy.optimize import minimize

from bilby_xG.motion import detector_position, sky_direction
from bilby_xG.networks import InterferometerList
from bilby_xG.utils import calculate_time_to_merger_for_any_mode

bilby.core.utils.logger.setLevel("WARNING")

#: GW170817-like, detector frame
SOURCE = dict(mass_1=1.48, mass_2=1.27, chi_1=0.0, chi_2=0.0, lambda_1=300.0,
              lambda_2=500.0, luminosity_distance=40.0, theta_jn=2.5, phase=0.0,
              psi=0.4)
GW170817 = dict(ra=3.44616, dec=-0.408084, geocent_time=1187008882.43)
GPS_2035 = 1735257618.0
YEAR = 365.25 * 86400
F_MAX = 2048.0
N_FREQ = 20_000

NETWORKS = {"ET (10 km triangle, from 3 Hz)": ["ET-EMR"],
            "CE 40 km + CE 20 km (from 5 Hz)": ["CE", "CE20"]}
#: the curvature of: the orbit alone (rotation in signal and templates),
#: the whole motion of the detectors
MOTIONS = {"orbit": "Orbit of the geocentre", "total": "Whole motion of the detectors"}
VARIANTS = ["Doppler at coalescence", "Optimal Doppler",
            "Optimal Doppler + chirp mass, mass ratio"]
# reference categorical palette, fixed order
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def polarizations(frequencies):
    return bilby.gw.source.binary_neutron_star_frequency_sequence(
        frequencies, mass_1=SOURCE["mass_1"], mass_2=SOURCE["mass_2"],
        luminosity_distance=SOURCE["luminosity_distance"], a_1=0.0, tilt_1=0.0,
        phi_12=0.0, a_2=0.0, tilt_2=0.0, phi_jl=0.0, lambda_1=SOURCE["lambda_1"],
        lambda_2=SOURCE["lambda_2"], theta_jn=SOURCE["theta_jn"], phase=SOURCE["phase"],
        waveform_approximant="IMRPhenomD_NRTidalv2", reference_frequency=50.0,
        frequencies=frequencies)


def trapezoid_weights(x):
    w = np.zeros_like(x)
    dx = np.diff(x)
    w[:-1] += dx / 2
    w[1:] += dx / 2
    return w


def deviation_from_uniform(vertex, tc, tau):
    """``R(t) - R(t_c) - V(t_c)(t - t_c)`` at ``t = t_c - tau`` (m), for the
    barycentric position of ``vertex`` (the geocentre if zero), in the axes
    of the sky position at ``t_c``."""
    position = detector_position(vertex, np.concatenate([tc - tau, [tc, tc - 1, tc + 1]]),
                                 axes_time=tc)
    velocity = (position[-1] - position[-2]) / 2
    return position[:-3] - position[-3] + velocity * tau[:, None]


class Network:
    """A network, its frequency grid and its SNR weights."""

    def __init__(self, names, minimum_frequency=None):
        self.ifos = InterferometerList(names)
        f_min = minimum_frequency or min(ifo.minimum_frequency for ifo in self.ifos)
        self.frequencies = np.geomspace(f_min, F_MAX, N_FREQ)
        pols = polarizations(self.frequencies)
        keep = np.abs(pols["plus"]) > 0
        self.frequencies = self.frequencies[keep]
        self.polarizations = {key: value[keep] for key, value in pols.items()}
        self.psds = [ifo.power_spectral_density.get_power_spectral_density_array(
            self.frequencies) for ifo in self.ifos]
        self.df = trapezoid_weights(self.frequencies)
        self.tau = calculate_time_to_merger_for_any_mode(
            self.frequencies, SOURCE["mass_1"], SOURCE["mass_2"], mode=2, safety=1)

    def weights(self, ra, dec, geocent_time):
        """``4 |h_ifo|^2 / S_ifo df``, shape (n_ifo, n_freq)."""
        parameters = dict(SOURCE, ra=ra, dec=dec, geocent_time=geocent_time)
        shared = {}
        out = []
        for ifo, psd in zip(self.ifos, self.psds):
            h = ifo.get_detector_response_for_frequency_dependent_antenna_response(
                self.polarizations, parameters, geocent_time - 1e5, self.frequencies,
                shared=shared)
            out.append(4 * np.abs(h) ** 2 / psd * self.df)
        return np.array(out)

    def dephasing(self, ra, dec, tc):
        """``2 pi f n.deviation/c`` for the orbit (common, shape (n_freq,))
        and for the whole motion of each detector (n_ifo, n_freq)."""
        direction = sky_direction(ra, dec)
        phase = 2 * np.pi * self.frequencies / speed_of_light
        orbit = phase * (deviation_from_uniform(np.zeros(3), tc, self.tau) @ direction)
        total = np.array([phase * (deviation_from_uniform(ifo.geometry.vertex, tc, self.tau)
                                   @ direction) for ifo in self.ifos])
        return orbit, total


def fit(weights, psi, basis, polish=False):
    """Maximise ``sum w exp(i (psi - basis @ c))`` over the coefficients
    ``c`` and a constant phase; returns (mismatch, residual phase, c). The
    arrays are flattened over detectors and frequencies. This is the
    weighted least-squares solution, exact for small residuals (polishing
    it by Nelder-Mead changes the mismatch by <1e-6 relative here)."""
    basis = [np.broadcast_to(b, psi.shape).ravel() for b in basis]
    weights, psi = weights.ravel(), psi.ravel()
    design = np.column_stack([np.ones_like(psi)] + basis)
    sw = np.sqrt(weights)
    # scale columns for conditioning
    scale = np.abs(design).max(axis=0)
    coefficients, *_ = np.linalg.lstsq(design / scale * sw[:, None], psi * sw, rcond=None)
    coefficients = coefficients / scale
    norm = weights.sum()

    def mismatch(c):
        residual = psi - design[:, 1:] @ c
        return 1 - np.abs(np.sum(weights * np.exp(1j * residual))) / norm

    c = coefficients[1:]
    if polish and len(c):
        result = minimize(lambda x: mismatch(x * scale[1:]), c / scale[1:],
                          method="Nelder-Mead",
                          options=dict(xatol=1e-12, fatol=1e-16, maxiter=4000))
        if result.fun < mismatch(c):
            c = result.x * scale[1:]
    residual = psi - design[:, 1:] @ c
    overlap = np.sum(weights * np.exp(1j * residual))
    residual = residual - np.angle(overlap)
    return 1 - np.abs(overlap) / norm, residual, c


def analyse(network, ra, dec, geocent_time):
    f, tau = network.frequencies, network.tau
    n_ifo = len(network.ifos)
    weights = network.weights(ra, dec, geocent_time)
    orbit, total = network.dephasing(ra, dec, geocent_time)
    time_shift = np.tile(2 * np.pi * f, (n_ifo, 1))
    doppler = np.tile(2 * np.pi * f * tau, (n_ifo, 1))
    newtonian = np.tile(f ** (-5 / 3), (n_ifo, 1))
    first_pn = np.tile(f ** (-1.), (n_ifo, 1))
    out = dict(snr2=weights.sum())
    for key, psi in (("orbit", np.tile(orbit, (n_ifo, 1))), ("total", total)):
        results = [fit(weights, psi, [time_shift]),
                   fit(weights, psi, [time_shift, doppler]),
                   fit(weights, psi, [time_shift, doppler, newtonian, first_pn])]
        out[f"mismatch_{key}"] = np.array([r[0] for r in results])
        # dephasing in the first detector: before and after the optimal fit
        out[f"dephasing_{key}"] = psi[0]
        out[f"residual_{key}"] = results[1][1].reshape(psi.shape)[0]
        out[f"beta_{key}"] = results[1][2][1]
    return out


def draws(n, seed=42):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(0, 2 * np.pi, n), np.arcsin(rng.uniform(-1, 1, n)),
                            GPS_2035 + rng.uniform(0, YEAR, n)])


def band(ax, x, samples, color, label, percentiles=(5, 95), ls="-"):
    low, median, high = np.percentile(samples, [percentiles[0], 50, percentiles[1]], axis=0)
    ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
    ax.plot(x, median, color=color, lw=2, label=label, ls=ls)


def displacement_figure(samples, outdir, network):
    tau = np.geomspace(1, 2 * 86400, 400)
    vertex = network.ifos[0].geometry.vertex
    curves = {"Orbit of the geocentre": [], "Rotation of the vertex": [],
              "Whole motion of the detector": []}
    for ra, dec, tc in samples:
        orbit = deviation_from_uniform(np.zeros(3), tc, tau)
        total = deviation_from_uniform(vertex, tc, tau)
        direction = sky_direction(ra, dec)
        for key, deviation in zip(curves, (orbit, total - orbit, total)):
            curves[key].append((np.linalg.norm(deviation, axis=1),
                                np.abs(deviation @ direction)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for k, (ax, title) in enumerate(zip(axes, ("Magnitude", "Along the line of sight"))):
        for (label, values), color in zip(curves.items(), COLORS):
            band(ax, tau / 3600, np.array([v[k] for v in values]), color, label)
        ax.plot(tau / 3600, 0.5 * 5.93e-3 * tau ** 2, color="black", ls=":", lw=1.5, zorder=5,
                label=r"$a_\oplus \tau^2 / 2$, $a_\oplus = GM_\odot/(1\,\mathrm{AU})^2$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-3, 3e8)
        ax.set_xlabel("Time before coalescence [h]")
        ax.grid(True, which="major", alpha=0.3)
        ax.set_title(title, fontsize=10, pad=34)
        _frequency_axis(ax, scale=1 / 3600)
    axes[0].set_ylabel("Displacement from uniform motion [m]")
    light = axes[1].secondary_yaxis("right", functions=(lambda x: x / speed_of_light * 1e3,
                                                        lambda x: x * speed_of_light / 1e3))
    light.set_ylabel("Light-travel time [ms]")
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle(f"{network.ifos[0].name} (ET, Euregio Meuse-Rhine): departure from its "
                 "velocity at coalescence (median, 5-95% over sky and time)", fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / "displacement.png", dpi=150)
    plt.close(fig)


def _frequency_axis(ax, scale):
    """Top axis: GW (2,2) frequency at that time before coalescence."""
    ticks = np.array([2, 3, 5, 10, 20, 50, 100, 1000])
    times = calculate_time_to_merger_for_any_mode(
        ticks, SOURCE["mass_1"], SOURCE["mass_2"], mode=2, safety=1) * scale
    lo, hi = ax.get_xlim()
    keep = (times > lo) & (times < hi)
    top = ax.secondary_xaxis("top")
    top.set_xticks(times[keep], [f"{f:g}" for f in ticks[keep]])
    top.set_xlabel("Frequency of the (2,2) mode [Hz]", fontsize=9)


def dephasing_figure(results, networks, outdir):
    fig, axes = plt.subplots(2, len(networks), figsize=(12, 8.5), sharey=True)
    series = [("dephasing_orbit", "Orbit, relative to its velocity at coalescence"),
              ("residual_orbit", "Orbit, residual after the optimal Doppler shift"),
              ("dephasing_total", "Whole motion, relative to the velocity at coalescence"),
              ("residual_total", "Whole motion, residual after the optimal Doppler shift")]
    for column, (name, network) in enumerate(networks.items()):
        rows = results[name]
        for row, (x, xlabel) in enumerate([
                (network.frequencies, "Frequency [Hz]"),
                (network.tau / 3600, "Time before coalescence [h]")]):
            ax = axes[row, column]
            for (key, label), color in zip(series, COLORS):
                band(ax, x, np.abs([r[key] for r in rows]), color, label)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(1e-7, 3)
            ax.set_xlabel(xlabel)
            ax.grid(True, which="major", alpha=0.3)
            if column == 0:
                ax.set_ylabel(f"|Dephasing| in {network.ifos[0].name} [rad]")
            if row == 0:
                ax.set_title(name, fontsize=10)
        # later times are clearer against frequency, above
        axes[1, column].set_xlim(network.tau[0] / 3600 * 1.2, 1e-3)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Dephasing from the departure of the detector motion from uniform "
                 "(median, 5-95% over sky and time)", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "dephasing.png", dpi=150)
    plt.close(fig)


def mismatch_figure(results, outdir, key):
    fig, axes = plt.subplots(2, len(results), figsize=(12, 7.5), sharey=True)
    for column, (name, rows) in enumerate(results.items()):
        mismatch = np.array([r[f"mismatch_{key}"] for r in rows])
        snr2 = np.array([r["snr2"] for r in rows])
        for row, (values, xlabel) in enumerate([
                (mismatch, "Mismatch $1 - \\mathcal{O}$"),
                (2 * snr2[:, None] * mismatch,
                 r"$\langle \delta h | \delta h \rangle \approx 2\rho^2 (1 - \mathcal{O})$")]):
            ax = axes[row, column]
            for k, (label, color) in enumerate(zip(VARIANTS, COLORS)):
                x = np.sort(np.clip(values[:, k], 1e-16, None))
                ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=color,
                        lw=2, label=label)
            ax.set_xscale("log")
            ax.set_xlabel(xlabel)
            ax.grid(True, which="major", alpha=0.3)
            if row == 1:
                ax.axvline(1, color="0.4", ls="--", lw=1)
            if column == 0:
                ax.set_ylabel("Fraction of sky positions and times")
            if row == 0:
                ax.set_title(f"{name}\nmedian network SNR {np.median(np.sqrt(snr2)):.0f}",
                             fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"{MOTIONS[key]}: signal against its uniform-motion approximation, "
                 "Doppler shifted", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / f"mismatch_{key}.png", dpi=150)
    plt.close(fig)


def fmin_figure(samples, outdir, minimum_frequencies=(2, 3, 4, 5, 7, 10, 15)):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    summary = {}
    for (name, names), color in zip(NETWORKS.items(), COLORS):
        # not below the lowest frequency of the noise curves
        lowest = min(ifo.power_spectral_density.frequency_array[0]
                     for ifo in InterferometerList(names))
        f_mins = [f for f in minimum_frequencies if f >= lowest]
        values = {key: [] for key in MOTIONS}
        for f_min in f_mins:
            network = Network(names, minimum_frequency=f_min)
            rows = [analyse(network, *s) for s in samples]
            for key in MOTIONS:
                values[key].append([r[f"mismatch_{key}"][1] for r in rows])
        short = name.split(" (")[0]
        for key, ls in zip(MOTIONS, ("-", "--")):
            v = np.array(values[key]).T
            band(ax, f_mins, v, color, f"{short}: {MOTIONS[key].lower()}", ls=ls)
            summary[f"{name}, {key}"] = dict(zip(map(str, f_mins),
                                                 np.median(v, axis=0).tolist()))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(minimum_frequencies, [str(f) for f in minimum_frequencies])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("Minimum frequency [Hz]")
    ax.set_ylabel("Mismatch to the optimal Doppler shift")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Mismatch vs low-frequency cutoff (median, 5-95%)", fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / "mismatch_vs_minimum_frequency.png", dpi=150)
    plt.close(fig)
    return summary


def main(outdir="orbital_motion_figures", n_samples=400):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)
    n_samples = int(n_samples)
    samples = draws(n_samples)
    networks = {name: Network(names) for name, names in NETWORKS.items()}

    results = {name: [analyse(network, *s) for s in samples]
               for name, network in networks.items()}
    summary = {}
    for name, rows in results.items():
        snr2 = np.array([r["snr2"] for r in rows])
        network = networks[name]
        event = analyse(network, GW170817["ra"], GW170817["dec"], GW170817["geocent_time"])
        entry = dict(duration_hours=float(network.tau[0] / 3600),
                     median_snr=float(np.median(np.sqrt(snr2))))
        for key in MOTIONS:
            mismatch = np.array([r[f"mismatch_{key}"] for r in rows])
            for k, v in enumerate(VARIANTS):
                entry[f"{key}: mismatch [{v}] median, 5%, 95%, max"] = np.percentile(
                    mismatch[:, k], [50, 5, 95, 100]).tolist()
                entry[f"{key}: 2 rho^2 mismatch [{v}] median, 95%, max"] = np.percentile(
                    2 * snr2 * mismatch[:, k], [50, 95, 100]).tolist()
            entry[f"{key}: median |optimal beta|"] = float(np.median(
                [abs(r[f"beta_{key}"]) for r in rows]))
            entry[f"{key}: GW170817 sky and time"] = dict(
                snr=float(np.sqrt(event["snr2"])),
                mismatch=dict(zip(VARIANTS, event[f"mismatch_{key}"].tolist())),
                max_abs_residual_optimal=float(np.abs(event[f"residual_{key}"]).max()))
        summary[name] = entry

    displacement_figure(samples[:200], outdir, networks[next(iter(networks))])
    dephasing_figure(results, networks, outdir)
    for key in MOTIONS:
        mismatch_figure(results, outdir, key)
    summary["median mismatch to the optimal Doppler shift vs minimum frequency"] = \
        fmin_figure(samples[:min(n_samples, 100)], outdir)

    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
