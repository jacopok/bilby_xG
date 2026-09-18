# Licensed under an MIT style license -- see LICENSE

"""A :class:`~bilby.gw.WaveformGenerator` that never caches an unused time
array.

bilby's own ``WaveformGenerator.frequency_domain_strain`` always evaluates
``self.time_array`` as an argument expression before calling
``_calculate_strain`` -- Python evaluates keyword-argument expressions
before the call itself -- even when only a ``frequency_domain_source_model``
is set (``time_domain_source_model=None``, the case for every source model
bilby_xG uses). ``_calculate_strain`` takes its ``model is not None`` branch
in that case and never actually reads ``transformed_model_data_points``, so
the access is pure waste: it triggers ``time_array``'s lazy full-band
(``duration * sampling_frequency``-sized) computation and caches it for the
lifetime of the object, for a value that is never used. At ET's low
minimum_frequency that is a ~2GB dead allocation per waveform generator,
picked up by every ``pickle.dump`` of anything holding one (a
relative-binning likelihood's ``.pickle`` on disk, a sampler's worker-pool
``initargs``) for no benefit. :meth:`time_domain_strain` has the same issue
in reverse (evaluates ``frequency_array`` unconditionally) -- overridden
symmetrically even though bilby_xG never calls it.
"""
from bilby.core import utils
from bilby.gw.waveform_generator import WaveformGenerator as _WaveformGenerator

__author__ = ["Pratyusava Baral <pbaral@uwm.edu>", "Soichiro Morisaki"]


class WaveformGenerator(_WaveformGenerator):
    def frequency_domain_strain(self, parameters=None):
        transformed_model_data_points = (
            self.time_array if self.time_domain_source_model is not None else None)
        return self._calculate_strain(
            model=self.frequency_domain_source_model,
            model_data_points=self.frequency_array,
            parameters=parameters,
            transformation_function=utils.nfft,
            transformed_model=self.time_domain_source_model,
            transformed_model_data_points=transformed_model_data_points)

    def time_domain_strain(self, parameters=None):
        transformed_model_data_points = (
            self.frequency_array if self.frequency_domain_source_model is not None else None)
        return self._calculate_strain(
            model=self.time_domain_source_model,
            model_data_points=self.time_array,
            parameters=parameters,
            transformation_function=utils.infft,
            transformed_model=self.frequency_domain_source_model,
            transformed_model_data_points=transformed_model_data_points)
