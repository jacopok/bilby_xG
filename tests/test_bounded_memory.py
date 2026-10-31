"""Signals of any length in bounded memory.

At a few-Hz lower cutoff a BNS lasts hours to a day, and the data's
frequency grid has ~1e8-1e9 points: any array of its size (the frequencies,
a band mask, a wide bin's samples, one per detector or worker process)
costs gigabytes. The relative-binning chain -- the injection as summary
data, the bin selection (with worker processes), the likelihood's summary
data, an evaluation and a pickle round trip -- walks the band a chunk at a
time (bilby_xG.grid.FrequencyGrid; a bin wider than a chunk in pieces), so
its peak memory must not grow with the duration: it takes longer, not more.
"""
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

TC = 1187008882.4 + 1e7
FMIN, FMAX = 20.0, 2048.0
SAMPLING_FREQUENCY = 4096
INJECTION = dict(
    chirp_mass=1.1975, mass_ratio=0.9, chi_1=0.01, chi_2=0.0, lambda_1=400.0,
    lambda_2=600.0, luminosity_distance=400.0, theta_jn=0.4, psi=1.0, phase=1.3,
    geocent_time=TC, ra=3.4, dec=-0.4)


def _waveform_generator(duration):
    from bilby_xG.source import convert_to_mlgw_bns_parameters, mlgw_bns_individual_modes
    from bilby_xG.waveform_generator import WaveformGenerator
    return WaveformGenerator(
        duration=duration, sampling_frequency=SAMPLING_FREQUENCY,
        frequency_domain_source_model=mlgw_bns_individual_modes,
        parameter_conversion=convert_to_mlgw_bns_parameters,
        waveform_arguments=dict(minimum_frequency=FMIN, maximum_frequency=FMAX,
                                mode_array=[[2, 2]]))


def _likelihood(duration, chunk_size, noise=False, bin_freqs=None, epsilon=0.1):
    from bilby_xG.injection import SummaryDataInjection
    from bilby_xG.likelihood import (
        RelativeBinningGravitationalWaveTransientNextGenerationModebyMode as Likelihood)
    from bilby_xG.networks import InterferometerList
    wfg = _waveform_generator(duration)
    ifos = InterferometerList(["ET-EMR"])
    data = SummaryDataInjection(wfg, INJECTION, TC + 2 - duration, FMIN, FMAX, noise=noise,
                                seed=3)
    data.setup_interferometers(ifos)
    return Likelihood(
        interferometers=ifos, waveform_generator=wfg, fiducial_parameters=INJECTION,
        bin_freqs=bin_freqs, epsilon=epsilon, time_reference="geocent", mode_array=[[2, 2]],
        injection=data, summary_data_chunk_size=chunk_size)


# -- the measurement, run in a fresh process per duration ---------------------

def _tree_pss(pid):
    """Proportional set size (bytes) of ``pid`` and its descendants: their
    memory, shared pages split between the processes sharing them (as a
    cgroup's limit counts them, unlike a sum of RSS)."""
    total, stack = 0, [pid]
    while stack:
        p = stack.pop()
        try:
            with open(f"/proc/{p}/smaps_rollup") as f:
                total += next(int(line.split()[1]) for line in f
                              if line.startswith("Pss:")) * 1024
            for task in os.listdir(f"/proc/{p}/task"):
                with open(f"/proc/{p}/task/{task}/children") as f:
                    stack += [int(c) for c in f.read().split()]
        except (OSError, StopIteration):
            pass
    return total


def _measure_chain(duration, chunk_size, npool):
    """Peak memory of the chain at ``duration``, a JSON line on stdout."""
    import pickle

    import bilby
    from bilby_xG.bin_selection import select_relative_binning_bins
    from bilby_xG.networks import InterferometerList
    bilby.core.utils.logger.setLevel("WARNING")

    peak, done = [0], [False]

    def sample():
        while not done[0]:
            peak[0] = max(peak[0], _tree_pss(os.getpid()))
            time.sleep(0.02)

    # the imports and the first waveform call (the surrogate's model) are
    # the same at any duration: the baseline
    _waveform_generator(duration).frequency_domain_source_model(
        np.array([FMIN, FMAX]), **_waveform_generator(duration).parameter_conversion(
            dict(INJECTION, fiducial=0))[0], frequency_bin_edges=np.array([FMIN, FMAX]),
        mode_array=[[2, 2]])
    baseline = _tree_pss(os.getpid())
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    t0 = time.time()

    priors = bilby.core.prior.PriorDict(dict(INJECTION))
    priors["mass_ratio"] = bilby.core.prior.Uniform(0.5, 1)
    priors["luminosity_distance"] = bilby.core.prior.Uniform(200, 800)
    priors["phase"] = bilby.core.prior.Uniform(0, 2 * np.pi)
    ifos = InterferometerList(["ET-EMR"])
    from bilby_xG.injection import SummaryDataInjection
    wfg = _waveform_generator(duration)
    data = SummaryDataInjection(wfg, INJECTION, TC + 2 - duration, FMIN, FMAX, noise=False)
    data.setup_interferometers(ifos)
    bilby.core.utils.random.seed(1)
    bin_freqs = select_relative_binning_bins(
        ifos, wfg, INJECTION, priors, FMIN, FMAX, eta=3.0, n_train=4, n_validation=2,
        n_evidence=8, n_scan_draws=1, scan_betas=np.geomspace(1e-4, 1, 3), n_grid=5001,
        npool=npool, evidence_tolerance=None)
    likelihood = _likelihood(duration, chunk_size, bin_freqs=bin_freqs)
    log_l = float(likelihood.log_likelihood_ratio(parameters=dict(INJECTION)))
    loaded = pickle.loads(pickle.dumps(likelihood))
    log_l_loaded = float(loaded.log_likelihood_ratio(parameters=dict(INJECTION)))
    done[0] = True
    thread.join()
    print(json.dumps(dict(
        duration=duration, frequencies=int(duration * SAMPLING_FREQUENCY / 2) + 1,
        baseline=baseline, peak=peak[0], seconds=time.time() - t0, log_l=log_l,
        log_l_loaded=log_l_loaded, n_bins=likelihood.number_of_bins)))


def _run_chain(duration, chunk_size, npool=2):
    import bilby_xG
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(
        [os.path.dirname(os.path.dirname(bilby_xG.__file__)), os.environ.get("PYTHONPATH", "")]))
    out = subprocess.run(
        [sys.executable, __file__, str(duration), str(chunk_size), str(npool)],
        capture_output=True, text=True, env=env, timeout=3600)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


# -- tests ----------------------------------------------------------------------

def test_frequency_grid_is_bilbys():
    from bilby.core.utils import create_frequency_series
    from bilby_xG.grid import FrequencyGrid
    rng = np.random.default_rng(0)
    for sampling_frequency, duration in [(4096, 4), (2048, 3.5), (8192, 1000.125),
                                         (8192, 65936)]:
        f = create_frequency_series(sampling_frequency, duration)
        grid = FrequencyGrid(sampling_frequency, duration)
        assert len(grid) == len(f)
        assert np.array_equal(grid.frequencies(), f)
        for a, b in [(0, 10), (len(f) - 7, len(f)), (len(f) // 3, len(f) // 3 + 999)]:
            assert np.array_equal(grid.frequencies(a, b), f[a:b])
        on_grid = f[rng.integers(0, len(f), 1000)]
        values = np.concatenate([
            rng.uniform(-1, sampling_frequency / 2 + 1, 1000), on_grid,
            np.nextafter(on_grid, 0), np.nextafter(on_grid, np.inf), [0, f[-1]]])
        for side in ("left", "right"):
            assert np.array_equal(grid.searchsorted(values, side),
                                  np.searchsorted(f, values, side))
        start, stop = grid.band(3.0, 1024.0)
        mask = (f >= 3.0) & (f <= 1024.0)
        assert (start, stop - start) == (np.argmax(mask), mask.sum())
        del f


def test_closed_form_bins_on_the_grid_a_chunk_at_a_time():
    from bilby_xG.grid import FrequencyGrid
    from bilby_xG.likelihood import relative_binning_bin_freqs, relative_binning_bin_freqs_on_grid
    grid = FrequencyGrid(4096, 512)
    for epsilon in (0.5, 0.05, 0.01):
        expected = relative_binning_bin_freqs(grid.frequencies(), FMIN, FMAX, epsilon=epsilon)
        for chunk_size in (2 ** 10, 2 ** 16, 2 ** 22):
            assert np.array_equal(relative_binning_bin_freqs_on_grid(
                grid, FMIN, FMAX, epsilon=epsilon, chunk_size=chunk_size), expected)


@pytest.mark.parametrize("noise", [False, True])
def test_bins_wider_than_a_chunk_in_pieces(noise):
    """Bins split into pieces (a chunk far smaller than the widest bins)
    give the summary data, and the noise draws, of whole bins: to rounding
    without noise; with noise to ~1e-7, the noise draw's sensitivity to the
    rounding of its bin's (here singular: the data are the fiducial) Gram
    matrix, whose terms the pieces sum in another order."""
    pytest.importorskip("mlgw_bns")
    whole = _likelihood(256, 2 ** 20, noise=noise)
    pieces = _likelihood(256, 2 ** 8, noise=noise)
    rtol = 1e-6 if noise else 1e-10
    widths = np.diff(whole.bin_inds)
    assert widths.max() > 2 ** 8 and widths.min() < 2 ** 8
    for name, data in whole.summary_data.items():
        for term, modes in data.items():
            for key, value in modes.items():
                values = value.values() if isinstance(value, dict) else [value]
                others = (pieces.summary_data[name][term][key].values()
                          if isinstance(value, dict) else [pieces.summary_data[name][term][key]])
                for a, b in zip(values, others):
                    np.testing.assert_allclose(b, a, rtol=rtol, atol=rtol * np.abs(a).max())
    assert pieces.noise_log_likelihood() == pytest.approx(whole.noise_log_likelihood(),
                                                          rel=rtol / 100)
    for parameters in (INJECTION, dict(INJECTION, chirp_mass=INJECTION["chirp_mass"] + 1e-5)):
        assert float(pieces.log_likelihood_ratio(parameters=dict(parameters))) == pytest.approx(
            float(whole.log_likelihood_ratio(parameters=dict(parameters))), abs=1e3 * rtol)


@pytest.mark.skipif(not os.path.exists("/proc/self/smaps_rollup"),
                    reason="needs Linux /proc to measure memory")
def test_memory_does_not_grow_with_the_duration():
    """8x the duration (8x the frequency samples, 1.7e7 to 1.3e8 bytes per
    float64 array of the grid): the peak memory of the whole chain, over the
    process and its bin-selection workers, grows by less than a quarter of
    one float64 array of the longer grid. Building any full-band array, once
    in any process, would exceed that."""
    pytest.importorskip("mlgw_bns")
    chunk_size = 2 ** 16
    short = _run_chain(1024, chunk_size)
    long = _run_chain(8192, chunk_size)
    growth = (long["peak"] - long["baseline"]) - (short["peak"] - short["baseline"])
    one_array = 8 * long["frequencies"]
    print(f"\npeak above baseline: {(short['peak'] - short['baseline']) / 2 ** 20:.0f} MB at "
          f"{short['duration']:g} s ({short['seconds']:.0f} s), "
          f"{(long['peak'] - long['baseline']) / 2 ** 20:.0f} MB at {long['duration']:g} s "
          f"({long['seconds']:.0f} s); one full-band float64 array: {one_array / 2 ** 20:.0f} MB")
    assert growth < one_array / 4
    for run in (short, long):
        assert run["log_l"] == run["log_l_loaded"] > 100


if __name__ == "__main__":
    _measure_chain(float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
