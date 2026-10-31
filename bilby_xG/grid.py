# Licensed under an MIT style license -- see LICENSE

"""The data's frequency grid, by index, without the array.

bilby's frequency grid (:func:`bilby.core.utils.create_frequency_series`)
is uniform: ``duration * sampling_frequency / 2 + 1`` points from 0 to the
Nyquist frequency. At a next-generation detector's few-Hz lower cutoff a
BNS lasts hours and the grid has ~1e8-1e9 points, so any array of its size
(the frequencies, a band mask, a masked copy, one per interferometer or per
worker process) costs gigabytes, and memory grows with the signal's length.
:class:`FrequencyGrid` gives the same frequencies, bit for bit, for any
index range, the in-band index range of a ``[minimum, maximum]`` band and
``numpy.searchsorted`` on the grid, so the relative-binning setup walks the
band a chunk at a time and its memory does not depend on the duration.
"""
import numpy as np

__all__ = ["FrequencyGrid"]


class FrequencyGrid:
    """bilby's frequency grid of ``duration`` and ``sampling_frequency``.

    ``frequencies(start, stop)`` equals
    ``create_frequency_series(sampling_frequency, duration)[start:stop]``
    exactly (``numpy.linspace``'s ``k * step``, the last point the Nyquist
    frequency itself).
    """

    def __init__(self, sampling_frequency, duration):
        number_of_samples = int(np.round(duration * sampling_frequency))
        self.size = int(np.round(number_of_samples / 2) + 1)
        self.maximum = sampling_frequency / 2
        self.step = self.maximum / (self.size - 1)
        self.duration = duration
        self.sampling_frequency = sampling_frequency

    @classmethod
    def of(cls, series):
        """The grid of anything with ``sampling_frequency`` and ``duration``
        (a waveform generator, an interferometer, its strain data)."""
        return cls(series.sampling_frequency, series.duration)

    def __len__(self):
        return self.size

    def __eq__(self, other):
        return (isinstance(other, FrequencyGrid) and self.size == other.size
                and self.maximum == other.maximum)

    def __repr__(self):
        return (f"FrequencyGrid(sampling_frequency={self.sampling_frequency}, "
                f"duration={self.duration}: {self.size} points)")

    def at(self, indices):
        """The frequencies at ``indices`` (array-like of ints in [0, size))."""
        indices = np.asarray(indices)
        out = indices * self.step
        return np.where(indices == self.size - 1, self.maximum, out)

    def frequencies(self, start=0, stop=None):
        """The frequencies of the indices ``start`` to ``stop`` (excluded)."""
        stop = self.size if stop is None else min(stop, self.size)
        start = max(start, 0)
        out = np.arange(start, max(stop, start), dtype=float) * self.step
        if stop == self.size and stop > start:
            out[-1] = self.maximum
        return out

    def searchsorted(self, values, side="left"):
        """``numpy.searchsorted(self.frequencies(), values, side)``."""
        values = np.asarray(values, dtype=float)
        k = np.clip(np.ceil(values / self.step), 0, self.size).astype(np.int64)
        # the estimate is off by at most one either way (rounding of
        # values / step and of k * step): settle it against the grid itself
        for _ in range(2):
            below = np.clip(k - 1, 0, self.size - 1)
            f_below = self.at(below)
            step_back = (k > 0) & ((f_below >= values) if side == "left" else (f_below > values))
            k = np.where(step_back, k - 1, k)
            here = np.clip(k, 0, self.size - 1)
            f_here = self.at(here)
            step_on = (k < self.size) & ((f_here < values) if side == "left" else (f_here <= values))
            k = np.where(step_on, k + 1, k)
        return k

    def band(self, minimum_frequency, maximum_frequency):
        """The index range ``(start, stop)`` of the frequencies in
        ``[minimum_frequency, maximum_frequency]`` (bilby's band mask)."""
        start = int(self.searchsorted(minimum_frequency, side="left"))
        stop = int(self.searchsorted(maximum_frequency, side="right"))
        return start, max(start, stop)

    def chunks(self, start, stop, chunk_size):
        """``(a, b)`` index ranges of at most ``chunk_size`` covering
        ``[start, stop)``."""
        for a in range(start, stop, chunk_size):
            yield a, min(a + chunk_size, stop)
