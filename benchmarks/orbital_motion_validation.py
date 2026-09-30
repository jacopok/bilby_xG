"""Validation of the cached (interpolated) Earth ephemeris of bilby_xG.motion.

The analogue of the ephemeris-caching validation of the LGWA response
(``lgwa_response.lunar_coordinates.make_position_interpolation_plot``), for
the barycentric motion of the geocentre. The exact reference is the ERFA
``epv00`` series itself, evaluated at random times over 2030-2040; the
cached ephemeris interpolates nodes spaced by ``step``, either linearly in
position (as the LGWA cache does) or by cubic Hermite interpolation of
positions and velocities (as :class:`bilby_xG.motion.EarthEphemeris` does).

The position error is converted to the phase error it causes at 2 kHz,
roughly the highest frequency of a BNS signal in a ground-based detector
(the LGWA plot used 3 Hz), assuming the worst-case orientation:
``2 pi f |dr| / c``.

Usage: python orbital_motion_validation.py [outdir]
"""
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from bilby.core.utils import speed_of_light

from bilby_xG.motion import (
    EarthEphemeris,
    earth_barycentric_position_velocity,
    hermite_interpolate,
    geocentre_delay,
)

GPS_2030, GPS_2040 = 1577491218.0, 1893024018.0
F_MAX = 2048.0
STEPS = np.array([60, 300, 600, 1800, 3600, 3 * 3600, 6 * 3600, 86400, 4 * 86400], dtype=float)
COLORS = {"linear": "#eb6834", "hermite": "#2a78d6"}


def interpolation_errors(steps, n_times=4000, seed=1):
    rng = np.random.default_rng(seed)
    times = rng.uniform(GPS_2030, GPS_2040, n_times)
    exact, _ = earth_barycentric_position_velocity(times)
    errors = {"linear": [], "hermite": []}
    for step in steps:
        # the two nodes around each time (the full ten-year table is
        # equivalent, only much larger for small steps)
        left = np.floor(times / step) * step
        nodes = np.stack([left, left + step])
        p, v = earth_barycentric_position_velocity(nodes)
        s = ((times - left) / step)[:, None]
        linear = (1 - s) * p[0] + s * p[1]
        hermite = np.array([
            hermite_interpolate(t, l_, step, pp, vv)
            for t, l_, pp, vv in zip(times, left, np.swapaxes(p, 0, 1), np.swapaxes(v, 0, 1))])
        errors["linear"].append(np.linalg.norm(linear - exact, axis=-1))
        errors["hermite"].append(np.linalg.norm(hermite - exact, axis=-1))
    return {key: np.array(value) for key, value in errors.items()}


def boxplot(ax, errors, steps):
    positions = np.arange(len(steps))
    for offset, (key, label) in zip((-0.18, 0.18), (("linear", "Linear in position"),
                                                   ("hermite", "Cubic Hermite (position + velocity)"))):
        box = ax.boxplot(errors[key].T, positions=positions + offset, widths=0.3,
                         patch_artist=True, whis=(5, 95), showfliers=False,
                         medianprops=dict(color="black"), label=label)
        for patch in box["boxes"]:
            patch.set_facecolor(COLORS[key])
            patch.set_edgecolor("black")
    ax.set_xticks(positions, [_step_label(s) for s in steps])
    ax.set_yscale("log")
    ax.set_xlabel("Node spacing")
    ax.set_ylabel("Position error [m]")
    ax.grid(True, axis="y", which="major", alpha=0.3)
    secondary = ax.secondary_yaxis(
        "right", functions=(lambda x: 2 * np.pi * F_MAX * x / speed_of_light,
                            lambda x: x * speed_of_light / (2 * np.pi * F_MAX)))
    secondary.set_ylabel(f"Max. phase error at {F_MAX:.0f} Hz [rad]")
    ax.legend(loc="upper left", frameon=False)


def _step_label(step):
    if step < 3600:
        return f"{step / 60:.0f} min"
    if step < 86400:
        return f"{step / 3600:.0f} h"
    return f"{step / 86400:.0f} d"


def relative_delay_errors(steps, n=300, duration=86400.0, seed=2):
    """Error of the orbital delay (relative to coalescence) over a one-day
    signal, as used in the response, against the exact series."""
    rng = np.random.default_rng(seed)
    ttc = np.geomspace(1e-2, duration, 400)
    out = []
    for step in steps:
        ephemeris = EarthEphemeris(step)
        errors = []
        for _ in range(n):
            tc = rng.uniform(GPS_2030, GPS_2040)
            ra, dec = rng.uniform(0, 2 * np.pi), np.arcsin(rng.uniform(-1, 1))
            cached = geocentre_delay(ra, dec, tc, ttc, ephemeris=ephemeris)
            exact = geocentre_delay(ra, dec, tc, ttc, ephemeris=_Exact())
            errors.append(np.abs(cached - exact).max())
        out.append(errors)
    return np.array(out)


class _Exact:
    """The series itself, with the interface of EarthEphemeris."""

    def position(self, times):
        return earth_barycentric_position_velocity(times)[0]

    def projected_position(self, times, direction):
        return self.position(times) @ direction


def main(outdir="orbital_motion_figures"):
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    errors = interpolation_errors(STEPS)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    boxplot(ax, errors, STEPS)
    ax.set_title("Interpolated vs exact barycentric position of the geocentre, 2030-2040\n"
                 "(boxes: quartiles; whiskers: 5-95%)", fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / "ephemeris_interpolation_error.png", dpi=150)
    plt.close(fig)

    delay_steps = STEPS[STEPS >= 600]
    delay_errors = relative_delay_errors(delay_steps)
    print("Orbital delay error over a 1-day signal, max over 300 draws "
          "(s; phase at 2 kHz, rad):")
    for step, e in zip(delay_steps, delay_errors):
        print(f"  step {_step_label(step):>6}: {e.max():.2e} s; "
              f"{2 * np.pi * F_MAX * e.max():.2e} rad")
    print("Position error, median / 95% (m):")
    for key in errors:
        for step, e in zip(STEPS, errors[key]):
            print(f"  {key:8s} {_step_label(step):>6}: {np.median(e):.2e} / "
                  f"{np.percentile(e, 95):.2e}")

    # cost of one evaluation at 1000 frequencies (bin edges), cached
    ephemeris = EarthEphemeris()
    ttc = np.geomspace(1e-3, 3e4, 1000)
    geocentre_delay(1.0, 0.2, 1.9e9, ttc, ephemeris=ephemeris)
    start = time.perf_counter()
    for _ in range(1000):
        geocentre_delay(1.0, 0.2, 1.9e9, ttc, ephemeris=ephemeris)
    per_call = (time.perf_counter() - start) / 1000
    start = time.perf_counter()
    earth_barycentric_position_velocity(1.9e9 - ttc)
    exact_call = time.perf_counter() - start
    print(f"Orbital delay at 1000 frequencies: {per_call * 1e6:.0f} us cached, "
          f"{exact_call * 1e6:.0f} us from the series")


if __name__ == "__main__":
    main(*sys.argv[1:])
