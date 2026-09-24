# Licensed under an MIT style license -- see LICENSE

"""Signal injections for long next-generation-detector signals.

At a 3 Hz lower cutoff a BNS in ET lasts hours, so the full frequency grid
has ~1e8 samples, and a single waveform call on it allocates several
temporaries of that size per mode. The injected signal is therefore always
built a chunk of frequencies at a time, through the relative-binning entry
point of a mode-by-mode source model (``fiducial=0`` with the
``frequency_bin_edges`` waveform argument), which every
``*_relative_binning_individual_modes`` model and
:func:`bilby_xG.source.mlgw_bns_individual_modes` accept.

:class:`SummaryDataInjection` is the default for simulated data. Passed to
:class:`bilby_xG.likelihood.RelativeBinningGravitationalWaveTransientNextGenerationModebyMode`
as ``injection``, it makes the likelihood build its relative-binning summary
data straight from the injected signal, plus a draw of their Gaussian-noise
contribution (or none, for a zero-noise injection). No full-band data
array is ever made. :func:`inject_zero_noise_chunked` instead sets the full
noiseless strain on the interferometers, for code that needs the data
themselves (e.g. an exact-likelihood comparison).
"""

import zlib

import numpy as np
from bilby.core.utils import logger
from bilby.gw.utils import noise_weighted_inner_product
from bilby.core.series import CoupledTimeAndFrequencySeries

#: Frequency samples per chunk; the same default as the relative-binning
#: likelihood's ``summary_data_chunk_size``.
DEFAULT_CHUNK_SIZE = 2 ** 20


class SummaryDataInjection:
    """A simulated signal (in Gaussian or zero noise) that exists only as
    relative-binning summary data.

    With data d = s + n, the likelihood's summary data are
    A_k(b) = <h0_k w_b | s> + <h0_k w_b | n>, with w_b = 1 and f - f_m(b).
    The signal term is integrated exactly over each bin; the noise terms of
    each bin are jointly circular complex Gaussian with covariance
    E[x_i x_j^*] = 2 <g_i | g_j> over the bin's basis functions g (all
    modes' h0_k and h0_k (f - f_m)), independent between bins and
    interferometers, as for stationary Gaussian noise with the
    interferometers' PSDs (Zackay, Dai & Venumadhav 2018). The real part of
    each noise term then has variance <g | g>.

    Parameters
    ----------
    waveform_generator : bilby_xG.waveform_generator.WaveformGenerator
        Its ``frequency_domain_source_model`` must be a mode-by-mode model
        that evaluates on ``frequency_bin_edges`` when ``fiducial=0``; its
        ``duration``, ``sampling_frequency`` and ``parameter_conversion``
        define the (never materialised) data grid and the conversion.
    parameters : dict
        Injection parameters (before ``parameter_conversion``).
    start_time : float
        GPS start time of the data segment.
    minimum_frequency, maximum_frequency : float
        The analysis band, set on the interferometers; the signal is zero
        below ``minimum_frequency`` (less one frequency bin).
    noise : bool
        Add a Gaussian-noise realisation (default) or inject in zero noise.
    seed : int, optional
        Seed of the noise realisation; drawn at random if not given (the
        drawn seed is logged and kept as ``self.seed``).
    earth_rotation_time_delay, earth_rotation_beam_patterns, finite_size : bool
        Detector-response options for the injected signal.
    """

    def __init__(self, waveform_generator, parameters, start_time,
                 minimum_frequency, maximum_frequency, noise=True, seed=None,
                 earth_rotation_time_delay=True, earth_rotation_beam_patterns=True,
                 finite_size=True):
        self.waveform_generator = waveform_generator
        self.parameters = dict(parameters)
        self.start_time = start_time
        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency
        self.noise = bool(noise)
        self.seed = int(np.random.SeedSequence(seed).entropy)
        self.earth_rotation_time_delay = bool(earth_rotation_time_delay)
        self.earth_rotation_beam_patterns = bool(earth_rotation_beam_patterns)
        self.finite_size = bool(finite_size)
        converted, _ = waveform_generator.parameter_conversion(
            dict(parameters, fiducial=0))
        self.converted_parameters = converted
        self.source_parameters = {key: converted[key]
                                  for key in waveform_generator.source_parameter_keys}
        logger.info(
            f"Injection as summary data, {'Gaussian' if self.noise else 'zero'} "
            f"noise{f' (seed {self.seed})' if self.noise else ''}: "
            f"{waveform_generator.frequency_domain_source_model.__name__} at {converted}")

    def setup_interferometers(self, interferometers):
        """Give the interferometers the data's time/frequency grid and band,
        without any strain array."""
        wfg = self.waveform_generator
        for ifo in interferometers:
            ifo.strain_data._frequency_domain_strain = None
            ifo.strain_data._time_domain_strain = None
            ifo.strain_data._times_and_frequencies = CoupledTimeAndFrequencySeries(
                duration=wfg.duration, sampling_frequency=wfg.sampling_frequency,
                start_time=self.start_time)
            ifo.minimum_frequency = self.minimum_frequency
            ifo.maximum_frequency = self.maximum_frequency

    def signal(self, interferometer, frequencies):
        """The injected signal in ``interferometer`` at ``frequencies``."""
        frequencies = np.asarray(frequencies)
        out = np.zeros(len(frequencies), dtype=complex)
        keep = frequencies > self.minimum_frequency - 1 / self.waveform_generator.duration
        if keep.any():
            out[keep] = _detector_signal(
                [interferometer], self.waveform_generator, self.source_parameters,
                self.converted_parameters, frequencies[keep], self.start_time,
                self.earth_rotation_time_delay, self.earth_rotation_beam_patterns,
                self.finite_size)[0]
        return out

    def noise_generator(self, interferometer):
        """The random generator of ``interferometer``'s noise realisation:
        fixed by ``self.seed`` and the name, whatever the order of the
        interferometers."""
        return np.random.default_rng(np.random.SeedSequence(
            self.seed, spawn_key=(zlib.crc32(interferometer.name.encode()),)))

    def optimal_snr_squared(self, interferometer, chunk_size=DEFAULT_CHUNK_SIZE):
        """<s|s> over the analysis band, in chunks."""
        freqs = _band(self.waveform_generator, self.minimum_frequency,
                      self.maximum_frequency)
        total = 0.0
        for i in range(0, len(freqs), chunk_size):
            f = freqs[i:i + chunk_size]
            s = self.signal(interferometer, f)
            psd = interferometer.power_spectral_density.get_power_spectral_density_array(f)
            total += noise_weighted_inner_product(
                s, s, psd, self.waveform_generator.duration).real
        return total


def complex_gaussian(covariance, z):
    """``L @ z`` with ``L L^H = covariance`` (Hermitian positive
    semi-definite; eigenvalues rounded below zero are clipped): a circular
    complex Gaussian vector with that covariance, given standard ones ``z``
    (E[z z^H] = I)."""
    w, v = np.linalg.eigh(covariance)
    return v @ (np.sqrt(np.clip(w, 0, None)) * z)


def _band(waveform_generator, minimum_frequency, maximum_frequency):
    freqs = waveform_generator.frequency_array
    return freqs[(freqs >= minimum_frequency) & (freqs <= maximum_frequency)]


def _detector_signal(interferometers, waveform_generator, source_parameters,
                     converted_parameters, frequencies, start_time,
                     earth_rotation_time_delay, earth_rotation_beam_patterns,
                     finite_size):
    polarizations = waveform_generator.frequency_domain_source_model(
        frequencies, **source_parameters,
        **dict(waveform_generator.waveform_arguments, frequency_bin_edges=frequencies))
    return [ifo.get_detector_response_for_frequency_dependent_antenna_response(
        waveform_polarizations=polarizations, parameters=converted_parameters,
        start_time=start_time, frequencies=frequencies,
        earth_rotation_time_delay=earth_rotation_time_delay,
        earth_rotation_beam_patterns=earth_rotation_beam_patterns,
        finite_size=finite_size) for ifo in interferometers]


def inject_zero_noise_chunked(interferometers, waveform_generator, parameters,
                              start_time, minimum_frequency, maximum_frequency,
                              chunk_size=DEFAULT_CHUNK_SIZE,
                              earth_rotation_time_delay=True,
                              earth_rotation_beam_patterns=True,
                              finite_size=True, progress=True):
    """Set each interferometer's strain to the noiseless signal.

    Only needed when the full data are; otherwise use
    :class:`SummaryDataInjection`.

    Parameters
    ----------
    interferometers : bilby_xG.networks.InterferometerList
    waveform_generator : bilby_xG.waveform_generator.WaveformGenerator
        Its ``frequency_domain_source_model`` must be a mode-by-mode model
        that evaluates on ``frequency_bin_edges`` when ``fiducial=0``; its
        ``frequency_array`` and ``parameter_conversion`` define the data grid
        and the conversion.
    parameters : dict
        Injection parameters (before ``parameter_conversion``).
    start_time : float
        GPS start time of the data segment.
    minimum_frequency, maximum_frequency : float
        The signal is zero below ``minimum_frequency`` (less one bin); the
        interferometers' analysis band is set to this range.

    Returns
    -------
    float
        The network optimal SNR.
    """
    frequency_array = waveform_generator.frequency_array
    df = frequency_array[1] - frequency_array[0]
    positions = np.flatnonzero(frequency_array > minimum_frequency - df)
    converted, _ = waveform_generator.parameter_conversion(dict(parameters, fiducial=0))
    # only the source model's own arguments, as the waveform generator passes
    source_parameters = {key: converted[key]
                         for key in waveform_generator.source_parameter_keys}
    source_model = waveform_generator.frequency_domain_source_model
    logger.info(f"Injecting {source_model.__name__} at {converted} in "
                f"{-(-len(positions) // chunk_size)} chunks")

    strain = {ifo.name: np.zeros_like(frequency_array, dtype=complex)
              for ifo in interferometers}
    chunks = range(0, len(positions), chunk_size)
    if progress:
        from tqdm import tqdm
        chunks = tqdm(chunks, desc="injection")
    for i in chunks:
        idx = positions[i:i + chunk_size]
        responses = _detector_signal(
            interferometers, waveform_generator, source_parameters, converted,
            frequency_array[idx], start_time, bool(earth_rotation_time_delay),
            bool(earth_rotation_beam_patterns), bool(finite_size))
        for ifo, response in zip(interferometers, responses):
            strain[ifo.name][idx] = response

    network_snr_squared = 0.0
    for ifo in interferometers:
        ifo.set_strain_data_from_frequency_domain_strain(
            strain.pop(ifo.name), start_time=start_time,
            frequency_array=frequency_array)
        ifo.minimum_frequency = minimum_frequency
        ifo.maximum_frequency = maximum_frequency
        snr_squared = ifo.optimal_snr_squared(signal=ifo.frequency_domain_strain).real
        logger.info(f"Injected optimal SNR in {ifo.name}: {np.sqrt(snr_squared):.1f}")
        network_snr_squared += snr_squared
    logger.info(f"Network optimal SNR: {np.sqrt(network_snr_squared):.1f}")
    return float(np.sqrt(network_snr_squared))
