"""How far from uniform is the detectors' motion in the Solar System over a
GW170817-like signal, and does it matter?

For a GW170817-like binary neutron star, over isotropic sky positions and
coalescence times uniform over a year, this computes

* the displacement of the geocentre from a uniform motion (its velocity
  at coalescence), ``r(t) - r(t_c) - v(t_c) (t - t_c)``, in 3D and along
  the line of sight;
* the dephasing it causes, as a function of frequency and of time to
  coalescence, relative to (a) the Doppler shift at coalescence,
  ``2 pi f [Delta(f) - n.v(t_c) tau(f) / c]``, and (b) the optimal
  Doppler shift, time and phase;
* the network mismatch between the signal with the orbital motion
  (``orbital_motion=True``) and the signal without it, optimally Doppler
  shifted.

To first order in v/c a uniform motion only multiplies all the
detector-frame masses (and the distance) by a Doppler factor, which to this
order is the same as adding ``beta * tau(f)`` to the arrival time, with
``tau`` the time to coalescence. The Doppler-shifted templates are
therefore the signal without orbital motion times
``exp(-2 pi i f (beta tau(f) + dt) + i phi)``; the mismatch is maximised
over ``phi`` analytically and over ``(beta, dt)`` numerically (from the
weighted-least-squares solution for the residual phase), with ``beta``
either fixed to the analytic ``n.v(t_c)/c`` or free. A third, linearised
variant also lets the chirp mass and mass ratio absorb what they can (the
Newtonian ``f^(-5/3)`` and 1PN ``f^(-1)`` phase terms). All the templates
share the detector response of the signal, so the residual phase is common
to all detectors and the network overlap is a sum over frequency of the
network ``sum_ifo |h_ifo|^2 / S_ifo`` weights.

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

from bilby_xG.networks import InterferometerList
from bilby_xG.orbit import (
    default_ephemeris,
    earth_barycentric_position_velocity,
    orbital_time_delay,
    source_direction,
)
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
VARIANTS = ["Doppler at coalescence", "Optimal Doppler",
            "Optimal Doppler + chirp mass, mass ratio"]
# reference categorical palette, fixed order
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


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


class Network:
    """The network SNR density of the signal (without orbital motion) at
    one sky position and time: ``4 sum_ifo |h_ifo|^2 / S_ifo df``."""

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

    def weights(self, ra, dec, geocent_time, orbital_motion=False):
        parameters = dict(SOURCE, ra=ra, dec=dec, geocent_time=geocent_time)
        shared = {}
        total = np.zeros_like(self.frequencies)
        for ifo, psd in zip(self.ifos, self.psds):
            h = ifo.get_detector_response_for_frequency_dependent_antenna_response(
                self.polarizations, parameters, geocent_time - 1e5, self.frequencies,
                orbital_motion=orbital_motion, shared=shared)
            total += 4 * np.abs(h) ** 2 / psd
        return total * self.df


def fit(weights, psi, basis, fixed=None):
    """Maximise sum_f w exp(i (psi - basis @ c)) over the coefficients c
    and a constant phase; returns (mismatch, residual phase, c).

    Starts from the weighted least-squares solution and polishes it."""
    target = psi if fixed is None else psi - fixed
    design = np.column_stack([np.ones_like(psi)] + basis)
    sw = np.sqrt(weights)
    # scale columns for conditioning
    scale = np.abs(design).max(axis=0)
    coefficients, *_ = np.linalg.lstsq(design / scale * sw[:, None], target * sw, rcond=None)
    coefficients = coefficients / scale
    norm = weights.sum()

    def mismatch(c):
        residual = target - design[:, 1:] @ c
        return 1 - np.abs(np.sum(weights * np.exp(1j * residual))) / norm

    c = coefficients[1:]
    if len(c):
        result = minimize(lambda x: mismatch(x * scale[1:]), c / scale[1:],
                          method="Nelder-Mead",
                          options=dict(xatol=1e-12, fatol=1e-16, maxiter=4000))
        if result.fun < mismatch(c):
            c = result.x * scale[1:]
    residual = target - design[:, 1:] @ c
    overlap = np.sum(weights * np.exp(1j * residual))
    residual = residual - np.angle(overlap)
    return 1 - np.abs(overlap) / norm, residual, c


def analyse(network, ra, dec, geocent_time):
    f, tau = network.frequencies, network.tau
    weights = network.weights(ra, dec, geocent_time)
    delay = orbital_time_delay(ra, dec, geocent_time, tau)
    psi = 2 * np.pi * f * delay
    _, velocity = earth_barycentric_position_velocity(geocent_time)
    beta = velocity @ source_direction(ra, dec) / speed_of_light
    doppler = 2 * np.pi * f * beta * tau
    out = dict(snr2=weights.sum(), beta=beta)
    out["mismatch_no_doppler"] = fit(weights, psi, [2 * np.pi * f])[0]
    out["dephasing_tc"] = psi - doppler
    results = [fit(weights, psi, [2 * np.pi * f], fixed=doppler),
               fit(weights, psi, [2 * np.pi * f, 2 * np.pi * f * tau]),
               fit(weights, psi, [2 * np.pi * f, 2 * np.pi * f * tau,
                                  f ** (-5 / 3), f ** (-1.)])]
    out["mismatch"] = np.array([r[0] for r in results])
    out["residual"] = np.array([r[1] for r in results])
    out["beta_optimal"] = results[1][2][1]
    return out


def draws(n, seed=42):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(0, 2 * np.pi, n), np.arcsin(rng.uniform(-1, 1, n)),
                            GPS_2035 + rng.uniform(0, YEAR, n)])


def band(ax, x, samples, color, label, percentiles=(5, 95)):
    low, median, high = np.percentile(samples, [percentiles[0], 50, percentiles[1]], axis=0)
    ax.fill_between(x, low, high, color=color, alpha=0.2, linewidth=0)
    ax.plot(x, median, color=color, lw=2, label=label)


def displacement_figure(samples, outdir, network):
    tau = np.geomspace(1, 2 * 86400, 400)
    magnitude, line_of_sight = [], []
    for ra, dec, tc in samples:
        position = default_ephemeris.position(np.append(tc - tau, tc))
        _, velocity = earth_barycentric_position_velocity(tc)
        deviation = position[:-1] - position[-1] + velocity * tau[:, None]
        magnitude.append(np.linalg.norm(deviation, axis=1))
        line_of_sight.append(np.abs(deviation @ source_direction(ra, dec)))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    band(ax, tau / 3600, np.array(magnitude), COLORS[0], "Magnitude")
    band(ax, tau / 3600, np.array(line_of_sight), COLORS[1], "Along the line of sight")
    ax.plot(tau / 3600, 0.5 * 5.93e-3 * tau ** 2, color="black", ls=":", lw=1.5, zorder=5,
            label=r"$a_\oplus \tau^2 / 2$, $a_\oplus = GM_\odot/(1\,\mathrm{AU})^2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, None)
    ax.set_xlabel("Time before coalescence [h]")
    ax.set_ylabel("Displacement from uniform motion [m]")
    light = ax.secondary_yaxis("right", functions=(lambda x: x / speed_of_light * 1e3,
                                                   lambda x: x * speed_of_light / 1e3))
    light.set_ylabel("Light-travel time [ms]")
    _frequency_axis(ax, network, scale=1 / 3600)
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Geocentre: departure from its velocity at coalescence "
                 "(median, 5-95% over sky and time)", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "displacement.png", dpi=150)
    plt.close(fig)


def _frequency_axis(ax, network, scale):
    """Top axis: GW (2,2) frequency at that time before coalescence."""
    ticks = np.array([f for f in (2, 3, 5, 10, 20, 50, 100, 1000)])
    times = calculate_time_to_merger_for_any_mode(
        ticks, SOURCE["mass_1"], SOURCE["mass_2"], mode=2, safety=1) * scale
    lo, hi = ax.get_xlim()
    keep = (times > lo) & (times < hi)
    top = ax.secondary_xaxis("top")
    top.set_xticks(times[keep], [f"{f:g}" for f in ticks[keep]])
    top.set_xlabel("Frequency of the (2,2) mode [Hz]")


def dephasing_figure(results, networks, outdir):
    fig, axes = plt.subplots(2, len(networks), figsize=(12, 8), sharey=True)
    for column, (name, network) in enumerate(networks.items()):
        rows = results[name]
        curves = [np.abs([r["dephasing_tc"] for r in rows])] + [
            np.abs([r["residual"][k] for r in rows]) for k in (1, 2)]
        labels = ["Relative to the Doppler shift at coalescence (not refitted)",
                  VARIANTS[1] + " (residual)", VARIANTS[2] + " (residual)"]
        for row, (x, xlabel, xscale) in enumerate([
                (network.frequencies, "Frequency [Hz]", "log"),
                (network.tau / 3600, "Time before coalescence [h]", "log")]):
            ax = axes[row, column]
            for curve, color, label in zip(curves, COLORS, labels):
                band(ax, x, curve, color, label)
            ax.set_xscale(xscale)
            ax.set_yscale("log")
            ax.set_ylim(1e-7, 10)
            ax.set_xlabel(xlabel)
            ax.grid(True, which="major", alpha=0.3)
            if column == 0:
                ax.set_ylabel("|Dephasing| [rad]")
            if row == 0:
                ax.set_title(name, fontsize=10)
        # later times are clearer against frequency, above
        axes[1, column].set_xlim(network.tau[0] / 3600 * 1.2, 1e-3)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Dephasing from the curvature of the orbital motion (median, 5-95% "
                 "over sky and time)", fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / "dephasing.png", dpi=150)
    plt.close(fig)


def mismatch_figure(results, outdir):
    fig, axes = plt.subplots(2, len(results), figsize=(12, 7.5), sharey=True)
    for column, (name, rows) in enumerate(results.items()):
        mismatch = np.array([r["mismatch"] for r in rows])
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
    fig.tight_layout()
    fig.savefig(outdir / "mismatch.png", dpi=150)
    plt.close(fig)


def fmin_figure(samples, outdir, minimum_frequencies=(2, 3, 4, 5, 7, 10, 15)):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    summary = {}
    for (name, names), color in zip(NETWORKS.items(), COLORS):
        # not below the lowest frequency of the noise curves
        lowest = min(ifo.power_spectral_density.frequency_array[0]
                     for ifo in InterferometerList(names))
        f_mins = [f for f in minimum_frequencies if f >= lowest]
        values = []
        for f_min in f_mins:
            network = Network(names, minimum_frequency=f_min)
            values.append([analyse(network, *s)["mismatch"][1] for s in samples])
        values = np.array(values).T
        band(ax, f_mins, values, color, name)
        summary[name] = dict(zip(map(str, f_mins), np.median(values, axis=0).tolist()))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(minimum_frequencies, [str(f) for f in minimum_frequencies])
    ax.set_xlabel("Minimum frequency [Hz]")
    ax.set_ylabel("Mismatch to the optimal Doppler shift")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(frameon=False)
    ax.set_title("Curvature mismatch vs low-frequency cutoff (median, 5-95%)", fontsize=10)
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
        mismatch = np.array([r["mismatch"] for r in rows])
        snr2 = np.array([r["snr2"] for r in rows])
        network = networks[name]
        event = analyse(network, GW170817["ra"], GW170817["dec"], GW170817["geocent_time"])
        summary[name] = dict(
            duration_hours=float(network.tau[0] / 3600),
            median_snr=float(np.median(np.sqrt(snr2))),
            mismatch_no_doppler_median=float(np.median([r["mismatch_no_doppler"] for r in rows])),
            **{f"mismatch [{v}] median, 5%, 95%, max": np.percentile(
                mismatch[:, k], [50, 5, 95, 100]).tolist() for k, v in enumerate(VARIANTS)},
            **{f"2 rho^2 mismatch [{v}] median, 95%, max": np.percentile(
                2 * snr2 * mismatch[:, k], [50, 95, 100]).tolist()
               for k, v in enumerate(VARIANTS)},
            beta_optimal_minus_beta_tc_median_abs=float(np.median(
                [abs(r["beta_optimal"] - r["beta"]) for r in rows])),
            GW170817_sky_and_time=dict(
                snr=float(np.sqrt(event["snr2"])),
                mismatch=dict(zip(VARIANTS, event["mismatch"].tolist())),
                max_abs_dephasing_optimal=float(np.abs(event["residual"][1]).max())))

    displacement_figure(samples[:200], outdir, networks[next(iter(networks))])
    dephasing_figure(results, networks, outdir)
    mismatch_figure(results, outdir)
    summary["median mismatch to the optimal Doppler shift vs minimum frequency"] = \
        fmin_figure(samples[:min(n_samples, 100)], outdir)

    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(*sys.argv[1:])
