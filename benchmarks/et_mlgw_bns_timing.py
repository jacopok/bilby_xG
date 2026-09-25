"""Timing breakdown of the mode-by-mode relative-binning likelihood for a BNS
in the Einstein Telescope (triangle, 3 interferometers) with the mlgw_bns
surrogate and its four modes (2,2), (2,1), (3,3), (4,4), from 3 Hz.

The (expensive, chunked) likelihood setup is cached in a pickle, so re-runs
only time the per-evaluation cost:

    python benchmarks/et_mlgw_bns_timing.py --cache et_mlgw.pickle

Reports the wall time of a likelihood evaluation, of the waveform alone (with
the numpy surrogate and, optionally, its JAX port), and of each stage of the
per-interferometer projection, then a cProfile of the full evaluation.
"""
import argparse
import cProfile
import os
import pickle
import pstats
import time
import warnings

import numpy as np

from bilby_xG.injection import SummaryDataInjection
from bilby_xG.interferometer import compute_wave_frame
from bilby_xG.likelihood import (
    RelativeBinningGravitationalWaveTransientNextGenerationModebyMode as Likelihood,
)
from bilby_xG.networks import InterferometerList
from bilby_xG.source import (MLGW_BNS_MODES, _mlgw_bns_model,
                             convert_to_mlgw_bns_parameters, mlgw_bns_individual_modes)
from bilby_xG.utils import calculate_time_to_merger_for_any_mode
from bilby_xG.waveform_generator import WaveformGenerator

FMIN, FMAX = 3.0, 2048.0
DURATION, SAMPLING_FREQUENCY = 2 ** 15, 4096
INJECTION = dict(
    chirp_mass=1.1975, mass_ratio=0.9, chi_1=0.01, chi_2=0.0, lambda_1=400.0,
    lambda_2=600.0, luminosity_distance=200.0, theta_jn=0.4, psi=1.0, phase=1.3,
    geocent_time=1187008882.4, ra=3.4, dec=-0.4)
START_TIME = INJECTION["geocent_time"] + 2 - DURATION


def build_likelihood(duration, sampling_frequency, epsilon):
    wfg = WaveformGenerator(
        duration=duration, sampling_frequency=sampling_frequency,
        frequency_domain_source_model=mlgw_bns_individual_modes,
        parameter_conversion=convert_to_mlgw_bns_parameters,
        waveform_arguments=dict(minimum_frequency=FMIN, maximum_frequency=FMAX,
                                mode_array=[list(mode) for mode in MLGW_BNS_MODES]))
    start_time = INJECTION["geocent_time"] + 2 - duration
    injection = SummaryDataInjection(wfg, INJECTION, start_time, FMIN, FMAX, noise=False)
    return Likelihood(
        interferometers=InterferometerList(["ET-EMR"]), waveform_generator=wfg,
        fiducial_parameters=INJECTION, epsilon=epsilon, time_reference="geocent",
        mode_array=[list(mode) for mode in MLGW_BNS_MODES], injection=injection)


def perturbed(rng, n):
    """Parameter sets near the injection, as a sampler would propose."""
    out = []
    for _ in range(n):
        p = dict(INJECTION)
        p["chirp_mass"] *= 1 + 1e-6 * rng.normal()
        p["mass_ratio"] = min(1.0, p["mass_ratio"] + 0.02 * rng.normal())
        for key in ("chi_1", "chi_2"):
            p[key] += 0.005 * rng.normal()
        for key in ("lambda_1", "lambda_2"):
            p[key] = max(0.0, p[key] + 50 * rng.normal())
        for key, scale in (("phase", 0.1), ("psi", 0.1), ("ra", 1e-2), ("dec", 1e-2),
                           ("theta_jn", 0.05), ("geocent_time", 1e-4),
                           ("luminosity_distance", 5.0)):
            p[key] += scale * rng.normal()
        out.append(p)
    return out


def timeit(function, samples, repeat=1):
    function(samples[0])  # warm up
    start = time.perf_counter()
    for _ in range(repeat):
        for sample in samples:
            function(sample)
    return (time.perf_counter() - start) / (repeat * len(samples))


def stage_timings(likelihood, samples):
    """Split one evaluation into the stages performed for each interferometer
    and mode, with the same calls as the likelihood makes."""
    wfg = likelihood.waveform_generator
    ifos = likelihood.interferometers
    freqs = likelihood.bin_freqs
    ms = 1e3
    rows = []

    def full(sample):
        return likelihood.log_likelihood_ratio(parameters=dict(sample))

    def waveform(sample):
        wfg._cache["parameters"] = None  # the generator caches the last call
        return wfg.frequency_domain_strain(dict(sample, fiducial=0))

    def converted(sample):
        p = dict(sample, fiducial=0)
        p.update(likelihood.get_sky_frame_parameters(p))
        return wfg.parameter_conversion(p)[0]

    rows.append(("log_likelihood_ratio (total)", timeit(full, samples) * ms))
    rows.append(("  waveform: frequency_domain_strain (4 modes, once)",
                 timeit(waveform, samples) * ms))
    rows.append(("  parameter_conversion (x1)", timeit(converted, samples) * ms))

    pols = [waveform(s) for s in samples]
    convs = [converted(s) for s in samples]
    pairs = list(zip(pols, convs))
    ifo = ifos[0]
    kwargs = dict(start_time=ifo.strain_data.start_time, frequencies=freqs,
                  earth_rotation_time_delay=True, earth_rotation_beam_patterns=True,
                  finite_size=True)

    def project_one_mode(pair):
        pol, conv = pair
        return ifo.get_detector_response_for_frequency_dependent_antenna_response(
            waveform_polarizations={"2,2": pol["2,2"]}, parameters=conv, **kwargs)

    def time_to_merger(pair):
        conv = pair[1]
        return calculate_time_to_merger_for_any_mode(
            freqs, conv["mass_1"], conv["mass_2"], conv["chi_1"], conv["chi_2"],
            mode=2, safety=1)

    ttc = time_to_merger(pairs[0])

    def frame(pair):
        conv = pair[1]
        return compute_wave_frame(conv["ra"], conv["dec"], conv["geocent_time"],
                                  conv["psi"], freqs, ttc)

    wave_frame = frame(pairs[0])

    def antenna(pair):
        conv = pair[1]
        return ifo.frequency_dependent_antenna_response(
            conv["ra"], conv["dec"], conv["geocent_time"], conv["psi"],
            times_to_coalescence=ttc, frequencies=freqs,
            start_time=kwargs["start_time"], wave_frame=wave_frame)

    def ratios(pair):
        return likelihood.compute_waveform_ratio_per_interferometer(
            pair[0], ifo, parameters=dict(samples[0], fiducial=0))

    def snrs(pair):
        return likelihood.calculate_snrs(pair[0], ifo, parameters=dict(samples[0], fiducial=0))

    n_ifo, n_mode = len(ifos), len(likelihood.mode_array)
    per_mode = timeit(project_one_mode, pairs) * ms
    rows.append((f"  calculate_snrs, one ifo (x{n_ifo})", timeit(snrs, pairs) * ms))
    rows.append((f"    compute_waveform_ratio, one ifo (x{n_ifo})", timeit(ratios, pairs) * ms))
    rows.append((f"      detector response, one mode, unshared (x{n_ifo * n_mode})", per_mode))
    rows.append((f"        time to merger (x{n_mode}, shared by the ifos)",
                 timeit(time_to_merger, pairs) * ms))
    rows.append((f"        wave frame (x{n_mode}, shared by the ifos)", timeit(frame, pairs) * ms))
    rows.append((f"        antenna response, given the frame (x{n_ifo * n_mode})",
                 timeit(antenna, pairs) * ms))
    return rows


def jax_waveform_timing(likelihood, samples):
    import jax
    from mlgw_bns.jax_predict import model_to_jax_waveform

    predict = jax.jit(model_to_jax_waveform(_mlgw_bns_model()))
    freqs = likelihood.bin_freqs
    conv = [convert_to_mlgw_bns_parameters(dict(s))[0] for s in samples]

    def call(p):
        q = 1 / p["q"] if p["q"] <= 1 else p["q"]
        hp, hc = predict(np.array([q, p["LambdaAl2"], p["LambdaBl2"], p["chi1z"], p["chi2z"]]),
                         freqs, p["M"], p["distance"], p["inclination"], p["coalescence_angle"])
        hp.block_until_ready()
        return hp, hc

    start = time.perf_counter()
    call(conv[0])
    compile_time = time.perf_counter() - start
    numpy_summed = timeit(lambda p: mlgw_bns_individual_modes(
        freqs, **{k: p[k] for k in ("M", "q", "chi1z", "chi2z", "LambdaAl2", "LambdaBl2",
                                    "distance", "inclination", "coalescence_angle")}), conv)
    return compile_time, timeit(call, conv), numpy_summed


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cache", default=None, help="pickle of the built likelihood")
    parser.add_argument("--duration", type=float, default=DURATION)
    parser.add_argument("--sampling-frequency", type=float, default=SAMPLING_FREQUENCY)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--jax", action="store_true", help="also time the JAX surrogate")
    parser.add_argument("--profile", default=None, help="write the cProfile stats here")
    args = parser.parse_args()
    # bilby >= 2.8 warns on every access of ``likelihood.parameters``
    warnings.simplefilter("ignore", FutureWarning)

    if args.cache and os.path.exists(args.cache):
        with open(args.cache, "rb") as f:
            likelihood = pickle.load(f)
    else:
        start = time.perf_counter()
        likelihood = build_likelihood(args.duration, args.sampling_frequency, args.epsilon)
        print(f"setup: {time.perf_counter() - start:.1f} s")
        if args.cache:
            with open(args.cache, "wb") as f:
                pickle.dump(likelihood, f)

    print(f"{len(likelihood.interferometers)} interferometers, "
          f"{len(likelihood.mode_array)} modes, {likelihood.number_of_bins} bins, "
          f"duration {likelihood.waveform_generator.duration:g} s")
    samples = perturbed(np.random.default_rng(1), args.n_samples)

    print(f"\n{'stage':<64} {'ms/call':>8}")
    for name, ms in stage_timings(likelihood, samples):
        print(f"{name:<64} {ms:8.3f}")

    if args.jax:
        compile_time, jax_ms, numpy_ms = jax_waveform_timing(likelihood, samples)
        print(f"\nwaveform on the {len(likelihood.bin_freqs)} bin edges:")
        print(f"  numpy mlgw_bns (per-mode)   {numpy_ms * 1e3:8.3f} ms")
        print(f"  jax mlgw_bns (summed, jit)  {jax_ms * 1e3:8.3f} ms "
              f"(compile {compile_time:.1f} s)")

    profiler = cProfile.Profile()
    profiler.enable()
    for sample in samples:
        likelihood.log_likelihood_ratio(parameters=dict(sample))
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("tottime")
    if args.profile:
        stats.dump_stats(args.profile)
    print(f"\ncProfile of {len(samples)} evaluations, by own time:")
    stats.print_stats(25)


if __name__ == "__main__":
    main()
