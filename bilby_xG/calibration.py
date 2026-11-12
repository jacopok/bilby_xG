# Licensed under an MIT style license -- see LICENSE

"""Detector calibration uncertainty in the relative-binning likelihoods.

The calibration factor of an interferometer multiplies the detector
response, h_cal(f) = C(f) h(f), with bilby's spline model
(:class:`bilby.gw.detector.calibration.CubicSpline`, LIGO-T2300140 Eq. 1):

    C(f) = (1 + dA(f)) (2 + i dphi(f)) / (2 - i dphi(f)),

dA and dphi cubic splines in log f through ``n_points`` nodes, the sampled
parameters ``{prefix}amplitude_{k}`` and ``{prefix}phase_{k}``. At fixed
frequencies dA and dphi are linear in the node values, so the spline is a
fixed matrix (:class:`CalibrationAtFrequencies`), the same values as
bilby's model without its parameter state. C is smooth over the nodes'
spacing, far wider than the relative-binning bins, so the likelihoods take
it at the bin edges as a factor of each mode's ratio to the fiducial
waveform (the fiducial is at the nominal calibration, C = 1).

A parameter set with none of an interferometer's calibration parameters is
at its nominal calibration (C = 1): the fiducial waveform, a search or a bin
selection that does not sample them. Some but not all of them is an error.
"""

import numpy as np

from bilby.gw.detector.calibration import CubicSpline, Recalibrate


class CalibrationAtFrequencies:
    """The spline calibration factor of ``model`` (a bilby ``CubicSpline``)
    at ``frequencies``, as a linear map of the node values: ``weights``
    (n_frequencies, n_points), with dA(f) = weights @ amplitude nodes."""

    def __init__(self, model, frequencies):
        frequencies = np.asarray(frequencies, dtype=float)
        n = model.n_points
        self.prefix = model.prefix
        self.n_points = n
        self.amplitude_keys = [f"{model.prefix}amplitude_{k}" for k in range(n)]
        self.phase_keys = [f"{model.prefix}phase_{k}" for k in range(n)]
        # as bilby's get_calibration_factor (the same clipping and spline)
        x = (np.log10(frequencies) - model.log_spline_points[0]) / model.delta_log_spline_points
        previous = np.clip(np.floor(x).astype(int), a_min=0, a_max=n - 2)
        b = x - previous
        a = 1 - b
        c = (a ** 3 - a) / 6
        d = (b ** 3 - b) / 6
        coefficients = np.asarray(model.nodes_to_spline_coefficients)
        rows = np.arange(len(frequencies))
        weights = c[:, None] * coefficients[previous] + d[:, None] * coefficients[previous + 1]
        weights[rows, previous] += a
        weights[rows, previous + 1] += b
        self.weights = weights
        self.frequencies = frequencies

    def nodes(self, parameters):
        """The (amplitude, phase) node values in ``parameters``, each
        ``(n_points,)`` or ``(n_points, N)``; None if it has none of them."""
        present = [key in parameters for key in self.amplitude_keys + self.phase_keys]
        if not any(present):
            return None
        if not all(present):
            missing = [key for key, here in zip(self.amplitude_keys + self.phase_keys, present)
                       if not here]
            raise KeyError(f"calibration parameters missing: {missing}")
        return (np.array([parameters[key] for key in self.amplitude_keys], dtype=float),
                np.array([parameters[key] for key in self.phase_keys], dtype=float))

    def factor(self, parameters):
        """C at the frequencies, ``(n_frequencies,)`` (or ``(n_frequencies,
        N)`` for arrays of N parameter sets); None at the nominal
        calibration (``parameters`` without calibration parameters)."""
        nodes = self.nodes(parameters)
        if nodes is None:
            return None
        return calibration_factor(self.weights @ nodes[0], self.weights @ nodes[1])


def calibration_factor(delta_amplitude, delta_phase):
    """bilby's calibration factor from the spline values."""
    return (1 + delta_amplitude) * (2 + 1j * delta_phase) / (2 - 1j * delta_phase)


def spline_model(interferometer):
    """The interferometer's calibration model if it is a spline
    (``CubicSpline``), None if it is the identity (bilby's ``Recalibrate``);
    other models are refused."""
    model = getattr(interferometer, "calibration_model", None)
    if model is None or type(model) is Recalibrate:
        return None
    if isinstance(model, CubicSpline):
        return model
    raise NotImplementedError(
        f"{interferometer.name}: the relative-binning likelihoods support bilby's "
        f"CubicSpline calibration model only, not {type(model).__name__}")


def calibration_at(interferometers, frequencies):
    """{name: CalibrationAtFrequencies or None} of each interferometer."""
    out = {}
    for ifo in interferometers:
        model = spline_model(ifo)
        out[ifo.name] = None if model is None else CalibrationAtFrequencies(model, frequencies)
    return out
