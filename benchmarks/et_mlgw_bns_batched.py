"""Batched vs numpy mode-by-mode relative-binning likelihood, for a BNS in the
Einstein Telescope with the mlgw_bns surrogate (the setup of
``et_mlgw_bns_timing.py``, with a shorter segment by default):

    python benchmarks/et_mlgw_bns_batched.py --duration 256 --fmin 20

Reports the cost per likelihood evaluation of the numpy likelihood and of
:class:`bilby_xG.batched.BatchedRelativeBinningLikelihood`, and their
log-likelihood differences. For scale, it also reports how much the numpy
likelihood itself changes under a relative perturbation of 1e-12 in
``lambda_1``: the surrogate's kernel-ridge regressors are ill-conditioned
(see :mod:`mlgw_bns.batched`), so their rounding error, which depends on the
batch size and on the backend, is a noise floor for any comparison.
"""
import argparse
import time
import warnings

import numpy as np

import et_mlgw_bns_timing as et
from bilby_xG.batched import BatchedRelativeBinningLikelihood


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--duration", type=float, default=256)
    parser.add_argument("--fmin", type=float, default=20.0)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    warnings.simplefilter("ignore", FutureWarning)

    et.FMIN = args.fmin
    start = time.perf_counter()
    likelihood = et.build_likelihood(args.duration, et.SAMPLING_FREQUENCY, args.epsilon)
    print(f"setup: {time.perf_counter() - start:.1f} s, {likelihood.number_of_bins} bins")

    samples = et.perturbed(np.random.default_rng(0), args.n_samples)
    start = time.perf_counter()
    reference = np.array([likelihood.log_likelihood_ratio(parameters=dict(s)) for s in samples])
    numpy_ms = (time.perf_counter() - start) / len(samples) * 1e3

    batched = BatchedRelativeBinningLikelihood(likelihood, batch_size=args.batch_size)
    parameters = {key: np.array([s[key] for s in samples]) for key in samples[0]}
    start = time.perf_counter()
    batched.log_likelihood_ratio(parameters)
    compile_s = time.perf_counter() - start
    start = time.perf_counter()
    result = batched.log_likelihood_ratio(parameters)
    batched_ms = (time.perf_counter() - start) / len(samples) * 1e3

    noise = []
    for sample in samples[:16]:
        nudged = dict(sample, lambda_1=sample["lambda_1"] * (1 + 1e-12))
        noise.append(likelihood.log_likelihood_ratio(parameters=nudged)
                     - likelihood.log_likelihood_ratio(parameters=dict(sample)))
    noise = np.abs(noise)

    difference = np.abs(result - reference)
    print(f"numpy likelihood     {numpy_ms:8.3f} ms per evaluation")
    print(f"batched likelihood   {batched_ms:8.3f} ms per evaluation "
          f"(batches of {args.batch_size}, compile {compile_s:.1f} s)")
    print(f"ln L ratio at the injection: numpy "
          f"{likelihood.log_likelihood_ratio(parameters=dict(et.INJECTION)):.4f}, batched "
          f"{batched.log_likelihood_ratio(dict(et.INJECTION)):.4f}")
    print(f"|batched - numpy| over {len(samples)} samples: median {np.median(difference):.3g}, "
          f"max {np.max(difference):.3g}")
    print(f"numpy |ln L(lambda_1 (1 + 1e-12)) - ln L|: median {np.median(noise):.3g}, "
          f"max {np.max(noise):.3g}")


if __name__ == "__main__":
    main()
