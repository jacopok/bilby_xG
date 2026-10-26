# Licensed under an MIT style license -- see LICENSE

"""Simulated data as a fixed realisation, and relative binning from it with
as few passes over the full-resolution data as possible.

An analysis from the data alone (no injection parameters, see
:mod:`bilby_xG.search`) looks at the data many times: a search, then the
summary data around the maximum-likelihood point. :class:`DataRealisation`
is the data: a signal plus Gaussian noise, the noise drawn per block of the
full frequency grid from the seed, so every pass over the data sees the same
realisation, and nothing at full resolution is kept between passes (each is
recomputed, chunk by chunk). Passed to
:class:`bilby_xG.likelihood.RelativeBinningGravitationalWaveTransientNextGenerationModebyMode`
as ``injection``, it makes the likelihood's summary data exact, from the
data.

One such pass, with very many bins (:func:`fine_likelihood`), stands for the
data near any fiducial: :func:`rebin` gives the summary data around another
fiducial, on fewer bins with edges among the fine ones, from the fine
summary data alone, by expanding the ratio of the two fiducials linearly
within each fine bin. The fine bins must be narrow enough for that ratio to
be linear across each of them, for every fiducial it is used for.
"""

import copy
import zlib

import numpy as np
from bilby.core.utils import logger

from .injection import SummaryDataInjection
from .likelihood import (
    RelativeBinningGravitationalWaveTransientNextGenerationModebyMode as Likelihood,
    _clear_waveform_cache, _log_likelihood_ratio_at)

#: samples of the full frequency grid per block of the noise realisation
NOISE_BLOCK = 2 ** 16


class DataRealisation(SummaryDataInjection):
    """A signal in a fixed Gaussian-noise realisation, at any frequencies of
    the data grid, from the seed alone.

    The noise at grid index i (frequency i / duration) is drawn by the block
    of :data:`NOISE_BLOCK` samples containing it, from ``seed``, the
    interferometer's name and the block's number, so it does not depend on
    which frequencies are asked for, in which order or chunks. Its real and
    imaginary parts have variance S T / 4 (bilby's normalisation, with
    E <n|n> = 2 per sample).

    To the likelihood this is a zero-noise injection (``noise`` is False):
    the strain it reads, :meth:`signal`, is the data. The parameters are
    those of :class:`bilby_xG.injection.SummaryDataInjection`, without
    ``noise``.
    """

    data = True

    def __init__(self, waveform_generator, parameters, start_time, minimum_frequency,
                 maximum_frequency, seed=None, **response):
        super().__init__(waveform_generator, parameters, start_time, minimum_frequency,
                         maximum_frequency, noise=False, **response)
        self.seed = int(np.random.SeedSequence(seed).entropy)
        logger.info(f"Data: the signal in Gaussian noise (seed {self.seed})")

    def signal(self, interferometer, frequencies):
        """The data in ``interferometer`` at ``frequencies``."""
        return (super().signal(interferometer, frequencies)
                + self.noise_at(interferometer, frequencies))

    def noise_at(self, interferometer, frequencies):
        """The noise realisation at ``frequencies`` (on the data grid)."""
        frequencies = np.asarray(frequencies, dtype=float)
        duration = self.waveform_generator.duration
        index = np.rint(frequencies * duration).astype(np.int64)
        order = np.argsort(index, kind="stable")
        blocks = index[order] // NOISE_BLOCK
        starts = np.flatnonzero(np.diff(blocks, prepend=-1))
        z = np.empty(len(index), dtype=complex)
        name = zlib.crc32(interferometer.name.encode())
        for start, end in zip(starts, np.append(starts[1:], len(index))):
            rng = np.random.default_rng(np.random.SeedSequence(
                self.seed, spawn_key=(name, int(blocks[start]))))
            draws = rng.standard_normal((NOISE_BLOCK, 2))
            within = index[order[start:end]] % NOISE_BLOCK
            z[order[start:end]] = draws[within, 0] + 1j * draws[within, 1]
        psd = interferometer.power_spectral_density.get_power_spectral_density_array(
            frequencies)
        noise = z * np.sqrt(np.where(np.isfinite(psd), psd, 0) * duration / 4)
        noise[frequencies < self.minimum_frequency] = 0
        return noise


def fine_likelihood(interferometers, waveform_generator, data, fiducial_parameters,
                    number_of_bins, priors=None, **kwargs):
    """The likelihood with ``number_of_bins`` bins geometric over the data's
    band (up to the fiducial's maximum frequency), its summary data from one
    pass over ``data`` (a :class:`DataRealisation`): the starting point of
    :func:`rebin`. ``kwargs`` go to the likelihood (the detector response,
    ``second_order``, ...)."""
    bin_freqs = np.geomspace(data.minimum_frequency, data.maximum_frequency,
                             number_of_bins + 1)
    return Likelihood(
        interferometers=interferometers, waveform_generator=waveform_generator,
        fiducial_parameters=fiducial_parameters, priors=priors, bin_freqs=bin_freqs,
        time_reference="geocent", injection=data, **kwargs)


def rebin(fine, fiducial_parameters, minimum_frequency=None, number_of_bins=None,
          bin_freqs=None):
    """The relative-binning likelihood around ``fiducial_parameters`` from
    the summary data of ``fine`` alone, without a pass over the data.

    Its bins are a subset of the fine ones: those nearest to
    ``number_of_bins`` geometric over [``minimum_frequency``, the lower of
    the two fiducials' maximum frequencies], or nearest to ``bin_freqs``.
    In each fine bin j (centre c_j) the ratio r_k = h0_k / h0_k^fine of each
    mode is linear, r_k = r0 + r1 (f - c_j) as in the likelihood, so that
    <h0_k (f - F)^n | d> and <h0_p (f - F)^n | h0_k (f - F)^m> over the
    coarse bin of centre F are sums of the fine a0, a1, b0, b1, b2 with
    coefficients from r0, r1 and c_j - F; the terms beyond the fine moments
    (of order r1 w_j^2 relative to r0 for fine width w_j) are dropped.
    """
    if fine.precessing_scheme != "ratio" or fine.legacy:
        raise ValueError("rebin needs precessing_scheme='ratio' and no legacy likelihood")
    new = copy.copy(fine)
    wfg = copy.copy(fine.waveform_generator)
    wfg.waveform_arguments = dict(fine.waveform_generator.waveform_arguments)
    new.waveform_generator = wfg
    new.__dict__.pop("_shared_response_cache", None)
    new.fiducial_parameters = dict(fiducial_parameters, fiducial=0)
    new.set_fiducial_waveforms(new.fiducial_parameters)

    edges = fine.bin_freqs
    lo = 0 if minimum_frequency is None else int(np.searchsorted(edges, minimum_frequency))
    hi = int(np.searchsorted(edges, min(new.maximum_frequency, edges[-1]), side="right")) - 1
    if bin_freqs is None:
        bin_freqs = np.geomspace(edges[lo], edges[hi], number_of_bins + 1)
    nearest = np.clip(np.searchsorted(edges, bin_freqs), lo + 1, hi)
    nearest -= (bin_freqs - edges[nearest - 1]) < (edges[nearest] - bin_freqs)
    index = np.unique(np.concatenate([[lo], nearest, [hi]]))
    index = index[(index >= lo) & (index <= hi)]
    segment = edges[lo:hi + 1]
    starts = index[:-1] - lo

    new.bin_freqs = edges[index]
    new.bin_inds = np.asarray(fine.bin_inds)[index]
    new.number_of_bins = len(index) - 1
    new.bin_widths = np.diff(new.bin_freqs)
    new.bin_centers = (new.bin_freqs[1:] + new.bin_freqs[:-1]) / 2
    wfg.waveform_arguments["frequency_bin_edges"] = new.bin_freqs
    _clear_waveform_cache(wfg)

    widths = np.diff(segment)
    centres = (segment[1:] + segment[:-1]) / 2
    coarse = np.searchsorted(index, np.arange(lo, hi), side="right") - 1
    delta = centres - new.bin_centers[coarse]
    keys = [f"{ell},{emm}" for ell, emm in new.mode_array]
    new.per_detector_per_mode_fiducial_waveform_points = {}
    new.summary_data = {}
    for ifo in new.interferometers:
        name = ifo.name
        r0, r1, points = {}, {}, {}
        for key in keys:
            h = new._reference_mode(ifo, key, segment)
            reference = fine.per_detector_per_mode_fiducial_waveform_points[name][key][lo:hi + 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(reference != 0, h / reference, 0)
            r0[key] = (ratio[1:] + ratio[:-1]) / 2
            r1[key] = np.diff(ratio) / widths
            points[key] = h[index - lo]
        new.per_detector_per_mode_fiducial_waveform_points[name] = points

        summary = fine.summary_data[name]
        fine_b2 = fine._second_order_summary_data(name)
        out = {kind: {} for kind in ("a0", "a1", "b0", "b1", "b2")}
        for key in keys:
            a0 = summary["a0"][key][lo:hi]
            a1 = summary["a1"][key][lo:hi]
            c0, c1 = np.conj(r0[key]), np.conj(r1[key])
            out["a0"][key] = np.add.reduceat(c0 * a0 + c1 * a1, starts)
            out["a1"][key] = np.add.reduceat(c0 * (a1 + delta * a0) + c1 * delta * a1, starts)
            for kind in ("b0", "b1", "b2"):
                out[kind][key] = {}
            for keyp in keys:
                # sum conj(h0_p) h0_k (f - c)^m conj(r_p) r_k (f - F)^n, with
                # conj(r_p) r_k = A + B u + C u^2, u = f - c, f - F = u + delta
                m0 = summary["b0"][key][keyp][lo:hi]
                m1 = summary["b1"][key][keyp][lo:hi]
                m2 = fine_b2[key][keyp][lo:hi]
                A = np.conj(r0[keyp]) * r0[key]
                B = np.conj(r0[keyp]) * r1[key] + np.conj(r1[keyp]) * r0[key]
                C = np.conj(r1[keyp]) * r1[key]
                out["b0"][key][keyp] = np.add.reduceat(A * m0 + B * m1 + C * m2, starts)
                out["b1"][key][keyp] = np.add.reduceat(
                    A * delta * m0 + (A + B * delta) * m1 + (B + C * delta) * m2, starts)
                out["b2"][key][keyp] = np.add.reduceat(
                    A * delta ** 2 * m0 + (2 * A * delta + B * delta ** 2) * m1
                    + (A + 2 * B * delta + C * delta ** 2) * m2, starts)
        new.summary_data[name] = out
    new.fiducial_likelihood = _log_likelihood_ratio_at(new, new.fiducial_parameters)
    return new


def overlaps(likelihood, parameters):
    """<d|h> (complex) and <h|h> of the waveform at ``parameters``, summed
    over the interferometers; (0, 1) where the waveform fails."""
    parameters = dict(parameters)
    polarizations = likelihood.waveform_generator.frequency_domain_strain(parameters)
    if polarizations is None:
        return 0j, 1.0
    parameters.update(likelihood.get_sky_frame_parameters(parameters))
    d_inner_h, h_inner_h = 0j, 0.0
    for ifo in likelihood.interferometers:
        snrs = likelihood.calculate_snrs(waveform_polarizations=polarizations,
                                         interferometer=ifo, return_array=False,
                                         parameters=parameters)
        d_inner_h += snrs.d_inner_h
        h_inner_h += snrs.optimal_snr_squared
    return d_inner_h, h_inner_h
