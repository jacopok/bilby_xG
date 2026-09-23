# Licensed under an MIT style license -- see LICENSE

"""Zero-noise signal injection for long next-generation-detector signals.

At a 3 Hz lower cutoff a BNS in ET lasts hours, so the full frequency grid
has ~1e8 samples, and a single waveform call on it allocates several
temporaries of that size per mode. :func:`inject_zero_noise_chunked` builds
the signal a chunk of frequencies at a time through the relative-binning
entry point of a mode-by-mode source model (``fiducial=0`` with the
``frequency_bin_edges`` waveform argument), which every
``*_relative_binning_individual_modes`` model and
:func:`bilby_xG.source.mlgw_bns_individual_modes` accept.
"""

import numpy as np
from bilby.core.utils import logger

#: Frequency samples per chunk; the same default as the relative-binning
#: likelihood's ``summary_data_chunk_size``.
DEFAULT_CHUNK_SIZE = 2 ** 20


def inject_zero_noise_chunked(interferometers, waveform_generator, parameters,
                              start_time, minimum_frequency, maximum_frequency,
                              chunk_size=DEFAULT_CHUNK_SIZE,
                              earth_rotation_time_delay=True,
                              earth_rotation_beam_patterns=True,
                              finite_size=True, progress=True):
    """Set each interferometer's strain to the noiseless signal.

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
        freqs = frequency_array[idx]
        polarizations = source_model(
            freqs, **source_parameters,
            **dict(waveform_generator.waveform_arguments, frequency_bin_edges=freqs))
        for ifo in interferometers:
            strain[ifo.name][idx] = ifo.get_detector_response_for_frequency_dependent_antenna_response(
                waveform_polarizations=polarizations, parameters=converted,
                start_time=start_time, frequencies=freqs,
                earth_rotation_time_delay=bool(earth_rotation_time_delay),
                earth_rotation_beam_patterns=bool(earth_rotation_beam_patterns),
                finite_size=bool(finite_size))

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
