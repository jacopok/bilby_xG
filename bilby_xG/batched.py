# Licensed under an MIT style license -- see LICENSE

"""Batched (vectorised) evaluation of the mode-by-mode relative-binning
likelihood, for samplers that propose many points at once (e.g. nessai).

Prototype. :class:`BatchedRelativeBinningLikelihood` wraps an existing
:class:`~bilby_xG.likelihood.RelativeBinningGravitationalWaveTransientNextGenerationModebyMode`
built with :func:`~bilby_xG.source.mlgw_bns_individual_modes`, and reuses
its bins, fiducial waveforms and summary data. Its
:meth:`~BatchedRelativeBinningLikelihood.log_likelihood_ratio` takes a
dict of parameter *arrays* and evaluates all of them in a single jitted,
``vmap``-ed JAX function: the mlgw_bns surrogate (through the JAX building
blocks shipped in :mod:`mlgw_bns.jax_predict`), the frequency-dependent
detector response and the summary-data contraction.

Scope of the prototype: general relativity only (no ``vG`` / ``(a, A)``),
no distance/phase/time marginalisation, ``reference_frame="sky"`` with
geocentre time. The JAX surrogate agrees with the numpy one to ~1e-4
in the inspiral; see ``benchmarks/et_mlgw_bns_batched.py`` for the
resulting log-likelihood differences and the speed-up.

:class:`BatchedBilbyModel` and :class:`BatchedNessai` connect it to nessai:
pass ``sampler=BatchedNessai`` to :func:`bilby.run_sampler`.
"""
import math

import numpy as np
from bilby.core.likelihood import Likelihood
from bilby.core.utils import logger, speed_of_light
from bilby_cython.geometry import greenwich_mean_sidereal_time

from .utils import calculate_time_to_merger_for_any_mode

__author__ = ["Jacopo Tissino"]

#: Solar mass in seconds, as used by mlgw_bns.
_SUN_MASS_SECONDS = 4.92549094830932e-6
#: mlgw_bns amplitude unit.
_AMP_SI_BASE = 4.2425873413901263e24
_DAY = 24. * 60. * 60.


def _ylm_iota(ell, emm, iota, xp):
    """``-2Y_lm(iota, 0)`` (real), the same convention as
    :func:`bilby_xG.source._spin_weighted_ylm`."""
    s = 2  # -(spin weight)
    c, s_ = xp.cos(iota / 2), xp.sin(iota / 2)
    norm = math.sqrt(math.factorial(ell + emm) * math.factorial(ell - emm)
                     * math.factorial(ell + s) * math.factorial(ell - s))
    out = 0.0
    for k in range(max(0, emm - s), min(ell + emm, ell - s) + 1):
        out = out + ((-1) ** k * c ** (2 * ell + emm - s - 2 * k) * s_ ** (2 * k + s - emm)
                     / (math.factorial(k) * math.factorial(ell + emm - k)
                        * math.factorial(ell - s - k) * math.factorial(s - emm + k)))
    return math.sqrt((2 * ell + 1) / (4 * math.pi)) * norm * out


def _mode_coefficients(ell, emm, iota, azimuth, xp):
    """The eight real coefficients of mlgw_bns' ``_build_mode_coeffs`` for
    one mode, with ``Y_lm(iota, phi) = Y_lm(iota, 0) exp(i m phi)``."""
    y, y_m = _ylm_iota(ell, emm, iota, xp), _ylm_iota(ell, -emm, iota, xp)
    cos, sin = xp.cos(emm * azimuth), xp.sin(emm * azimuth)
    yr, yi = y * cos, y * sin
    yr_m, yi_m = y_m * cos, -y_m * sin
    if ell % 2:
        return (yr - yr_m, -(yi + yi_m), yi + yi_m, yr - yr_m,
                -(yi - yi_m), -(yr + yr_m), yr + yr_m, -(yi - yi_m))
    return (yr + yr_m, -(yi - yi_m), yi - yi_m, yr + yr_m,
            -(yi + yi_m), -(yr - yr_m), yr - yr_m, -(yi + yi_m))


def mlgw_bns_jax_modes(model, mode_array):
    """A JAX function evaluating selected mlgw_bns modes for one parameter set.

    Returns ``modes(theta, total_mass, distance, inclination, phase,
    frequencies) -> (plus, cross)``, each of shape ``(n_modes, n_freqs)``,
    with ``theta = [q >= 1, lambda_1, lambda_2, chi_1, chi_2]``: the same
    per-mode polarisations as :func:`bilby_xG.source.mlgw_bns_individual_modes`
    (up to the JAX port's ~1e-4 accuracy). Use with ``jax.vmap``.
    """
    import jax.numpy as jnp
    from mlgw_bns.higher_order_modes import Mode
    from mlgw_bns.jax_predict import (
        _mode_pn_amp, _mode_pn_phase, make_not_a_knot_spline_jax,
        mode_model_to_jax_residuals, mode_phases_nn_to_jax, timeshifts_nn_to_jax)

    dataset = model.dataset
    reference_mass = float(dataset.total_mass)
    connection_hz = float(dataset.effective_initial_frequency_hz)
    connection_natural = connection_hz * float(dataset.mass_sum_seconds)
    reference = dataset.amplitude_reference_parameters
    if reference is None:
        raise NotImplementedError("needs a model with a frozen PN amplitude reference")
    reference_eta = reference.mass_ratio / (1 + reference.mass_ratio) ** 2
    reference_chi_a = (reference.chi_1 - reference.chi_2) / 2
    reference_chi_s = (reference.chi_1 + reference.chi_2) / 2

    phases_predictor = model.mode_phases_predictor
    mode_phases = mode_phases_nn_to_jax(phases_predictor, model.modes)
    phase_columns = [tuple(mode) for mode in phases_predictor.modes]
    time_shifts = timeshifts_nn_to_jax(model.time_shifts_predictor)

    per_mode = []
    for ell, emm in mode_array:
        mode_model = model.mode_models[Mode(ell, emm)]
        indices = mode_model.downsampling_indices
        hz = np.asarray(mode_model.dataset.frequencies_hz)
        natural = np.asarray(mode_model.dataset.frequencies)
        per_mode.append(dict(
            lm=(ell, emm),
            residuals=mode_model_to_jax_residuals(mode_model),
            n_amp=indices.amp_length,
            amp_spline=make_not_a_knot_spline_jax(hz[indices.amplitude_indices]),
            phi_spline=make_not_a_knot_spline_jax(hz[indices.phase_indices]),
            phi_natural=jnp.asarray(natural[indices.phase_indices]),
            pn_amp=_mode_pn_amp((ell, emm), jnp.asarray(natural[indices.amplitude_indices]),
                                reference_eta, reference_chi_a, reference_chi_s),
            phase_column=phase_columns.index((ell, emm)),
            max_hz=float(hz[-1]),
        ))

    def modes(theta, total_mass, distance, inclination, phase, frequencies):
        q, lambda_1, lambda_2, chi_1, chi_2 = theta
        eta = q / (1 + q) ** 2
        chi_a, chi_s = (chi_1 - chi_2) / 2, (chi_1 + chi_2) / 2
        row = theta[None, :]
        mass_rescaling = total_mass / reference_mass
        rescaled = frequencies * mass_rescaling
        natural = frequencies * total_mass * _SUN_MASS_SECONDS
        # post-Newtonian extension below the trained band, blended in over
        # [connection / 2, connection] (mlgw_bns' low-frequency splice)
        low = rescaled < connection_hz
        blend = jnp.clip((rescaled - connection_hz / 2) / (connection_hz / 2), 0.0, 1.0)
        blend = (1 - jnp.cos(math.pi * blend)) / 2
        time_shift = time_shifts(row)[0] * mass_rescaling
        time_shift_phase = 2 * math.pi * (frequencies - connection_hz / mass_rescaling) * time_shift
        phase0 = mode_phases(row)[0]
        azimuth = math.pi / 2 - phase
        prefactor = total_mass ** 2 / _AMP_SI_BASE / distance / 2

        plus, cross = [], []
        for m in per_mode:
            lm = m["lm"]
            residuals = m["residuals"](row)[0]
            amp_nodes = m["pn_amp"] * residuals[:m["n_amp"]]
            phi_nodes = (_mode_pn_phase(lm, m["phi_natural"], eta, chi_1, chi_2, chi_a, chi_s,
                                        lambda_1, lambda_2)
                         + residuals[m["n_amp"]:] + phase0[m["phase_column"]])
            amp = m["amp_spline"](amp_nodes, rescaled)
            phi = m["phi_spline"](phi_nodes, rescaled)

            amp_connection = m["amp_spline"](amp_nodes, jnp.asarray(connection_hz))
            phi_connection = m["phi_spline"](phi_nodes, jnp.asarray(connection_hz))
            natural_connection = jnp.asarray([connection_natural])
            low_amp = _mode_pn_amp(lm, natural, eta, chi_a, chi_s)
            low_phi = _mode_pn_phase(lm, natural, eta, chi_1, chi_2, chi_a, chi_s,
                                     lambda_1, lambda_2)
            low_amp_connection = _mode_pn_amp(lm, natural_connection, eta, chi_a, chi_s)[0]
            low_phi_connection = _mode_pn_phase(
                lm, natural_connection, eta, chi_1, chi_2, chi_a, chi_s, lambda_1, lambda_2)[0]
            amp = jnp.where(low, low_amp + blend * (amp_connection - low_amp_connection), amp)
            phi = jnp.where(low, low_phi + (phi_connection - low_phi_connection), phi)
            amp = jnp.where(rescaled > m["max_hz"], 0.0, amp) * prefactor
            phi = phi + time_shift_phase

            c = _mode_coefficients(*lm, inclination, azimuth, jnp)
            cos, sin = jnp.cos(phi), jnp.sin(phi)
            plus.append(amp * ((cos * c[0] + sin * c[1]) + 1j * (cos * c[2] + sin * c[3])))
            cross.append(amp * ((cos * c[4] + sin * c[5]) + 1j * (cos * c[6] + sin * c[7])))
        positive = frequencies > 0
        return (jnp.where(positive, jnp.stack(plus), 0.0),
                jnp.where(positive, jnp.stack(cross), 0.0))

    return modes


def _antenna_response(geometry, frame, frequencies, ifo_time, flags):
    """``Interferometer.frequency_dependent_antenna_response`` (general
    relativity) in JAX, for one detector, given the wave frame."""
    import jax.numpy as jnp

    earth_rotation_time_delay, earth_rotation_beam_patterns, finite_size = flags
    omegas, pol_plus, pol_cross = frame

    def constant(x):
        return x if earth_rotation_beam_patterns else jnp.full_like(x, x[-1])

    if finite_size:
        fpxx = constant(jnp.einsum('ij,ijk->k', geometry["xx"], pol_plus))
        fpyy = constant(jnp.einsum('ij,ijk->k', geometry["yy"], pol_plus))
        fcxx = constant(jnp.einsum('ij,ijk->k', geometry["xx"], pol_cross))
        fcyy = constant(jnp.einsum('ij,ijk->k', geometry["yy"], pol_cross))
        fl_over_c = frequencies * geometry["length"] * 1e3 / speed_of_light

        def arm(y):
            return 0.5 * (jnp.exp(-1j * math.pi * fl_over_c * (1 + y)) * jnp.sinc(fl_over_c * (1 - y))
                          + jnp.exp(1j * math.pi * fl_over_c * (1 - y)) * jnp.sinc(fl_over_c * (1 + y)))

        dxx = arm(-omegas.T @ geometry["x"])
        dyy = arm(-omegas.T @ geometry["y"])
        fps = fpxx * dxx - fpyy * dyy
        fcs = fcxx * dxx - fcyy * dyy
    else:
        fps = constant(jnp.einsum('ij,ijk->k', geometry["detector_tensor"], pol_plus))
        fcs = constant(jnp.einsum('ij,ijk->k', geometry["detector_tensor"], pol_cross))

    ifo_times = ifo_time - omegas.T @ geometry["vertex"] / speed_of_light
    if not earth_rotation_time_delay:
        ifo_times = ifo_times[-1]
    exp_fac = jnp.exp(-2j * math.pi * frequencies * ifo_times)
    return fps * exp_fac, fcs * exp_fac


def _wave_frame(ra, dec, psi, gmsts):
    """:func:`bilby_xG.interferometer.compute_wave_frame` in JAX."""
    import jax.numpy as jnp

    theta, phi = jnp.pi / 2 - dec, ra - gmsts
    cos_phi, sin_phi = jnp.cos(phi), jnp.sin(phi)
    cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)
    zero = jnp.zeros_like(phi)
    u = jnp.stack([cos_phi * cos_theta, cos_theta * sin_phi, -sin_theta * jnp.ones_like(phi)])
    v = jnp.stack([-sin_phi, cos_phi, zero])
    m = -u * jnp.sin(psi) - v * jnp.cos(psi)
    n = -u * jnp.cos(psi) + v * jnp.sin(psi)
    omegas = jnp.stack([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta * jnp.ones_like(phi)])
    mn = jnp.einsum('ik,jk->ijk', m, n)
    pol_plus = jnp.einsum('ik,jk->ijk', m, m) - jnp.einsum('ik,jk->ijk', n, n)
    pol_cross = mn + jnp.transpose(mn, (1, 0, 2))
    return omegas, pol_plus, pol_cross


class BatchedRelativeBinningLikelihood(Likelihood):
    """Vectorised version of a mode-by-mode relative-binning likelihood.

    Parameters
    ----------
    likelihood : RelativeBinningGravitationalWaveTransientNextGenerationModebyMode
        A set-up likelihood whose source model is
        :func:`~bilby_xG.source.mlgw_bns_individual_modes`; its summary data,
        bins and fiducial waveforms are reused as they are.
    batch_size : int
        Points are evaluated in chunks of this many (the last one padded):
        each distinct chunk size is compiled once (tens of seconds), so
        only two shapes are ever used, this and 1 (for single points).
    """

    def __init__(self, likelihood, batch_size=256):
        import jax

        from .source import _mlgw_bns_model, mlgw_bns_individual_modes

        super().__init__()
        self.likelihood = likelihood
        self.batch_size = int(batch_size)
        wfg = likelihood.waveform_generator
        if wfg.frequency_domain_source_model is not mlgw_bns_individual_modes:
            raise NotImplementedError("the batched likelihood supports mlgw_bns_individual_modes only")
        if (likelihood.distance_marginalization or likelihood.phase_marginalization
                or likelihood.time_marginalization):
            raise NotImplementedError("marginalisation is not supported by the batched likelihood")
        if likelihood.reference_frame != "sky" or "geocent" not in likelihood.time_reference:
            raise NotImplementedError("only reference_frame='sky' with geocentre time is supported")

        self.mode_array = [tuple(int(x) for x in mode) for mode in likelihood.mode_array]
        keys = [f"{ell},{emm}" for ell, emm in self.mode_array]
        ifos = likelihood.interferometers
        self.frequencies = np.asarray(likelihood.bin_freqs, dtype=float)
        self.start_time = ifos[0].strain_data.start_time
        if any(ifo.strain_data.start_time != self.start_time for ifo in ifos):
            raise NotImplementedError("the interferometers must share a start time")

        summary = likelihood.summary_data
        stack = np.stack
        self._arrays = dict(
            a0=stack([stack([summary[ifo.name]["a0"][k] for k in keys]) for ifo in ifos]),
            a1=stack([stack([summary[ifo.name]["a1"][k] for k in keys]) for ifo in ifos]),
            b0=stack([stack([stack([summary[ifo.name]["b0"][k][kp] for kp in keys]) for k in keys])
                      for ifo in ifos]),
            b1=stack([stack([stack([summary[ifo.name]["b1"][k][kp] for kp in keys]) for k in keys])
                      for ifo in ifos]),
            reference=stack([stack([likelihood.per_detector_per_mode_fiducial_waveform_points[
                ifo.name][k] for k in keys]) for ifo in ifos]),
            bin_widths=np.asarray(likelihood.bin_widths, dtype=float),
            frequencies=self.frequencies,
        )
        self._geometries = [dict(
            xx=np.asarray(ifo.geometry.xx), yy=np.asarray(ifo.geometry.yy),
            detector_tensor=np.asarray(ifo.geometry.detector_tensor),
            x=np.asarray(ifo.geometry.x), y=np.asarray(ifo.geometry.y),
            vertex=np.asarray(ifo.geometry.vertex), length=float(ifo.geometry.length),
        ) for ifo in ifos]
        self._flags = (bool(likelihood.earth_rotation_time_delay),
                       bool(likelihood.earth_rotation_beam_patterns),
                       bool(likelihood.finite_size))
        self._modes = mlgw_bns_jax_modes(_mlgw_bns_model(), self.mode_array)
        self._batched = jax.jit(jax.vmap(self._single, in_axes=(0, None)))

    def _single(self, p, arrays):
        """ln L ratio for one parameter set (a dict of scalars)."""
        import jax.numpy as jnp

        f = arrays["frequencies"]
        theta = jnp.stack([p["q"], p["lambda_1"], p["lambda_2"], p["chi_1"], p["chi_2"]])
        plus, cross = self._modes(theta, p["total_mass"], p["distance"], p["inclination"],
                                  p["phase"], f)
        ifo_time = p["geocent_time"] - self.start_time
        earth_rotation = self._flags[0] or self._flags[1]

        frames = {}
        ratios = []
        for geometry in self._geometries:
            strains = []
            for k, (_, emm) in enumerate(self.mode_array):
                if emm not in frames:
                    ttc = calculate_time_to_merger_for_any_mode(
                        f, p["mass_1"], p["mass_2"], p["chi_1"], p["chi_2"], mode=emm, safety=1)
                    gmsts = (p["gmst"] - p["gmst_rate"] * ttc if earth_rotation
                             else jnp.full_like(f, p["gmst"]))
                    frames[emm] = _wave_frame(p["ra"], p["dec"], p["psi"], gmsts)
                fps, fcs = _antenna_response(geometry, frames[emm], f, ifo_time, self._flags)
                strains.append(plus[k] * fps + cross[k] * fcs)
            ratios.append(jnp.stack(strains))
        ratio = jnp.stack(ratios) / arrays["reference"]  # (ifo, mode, edge)
        r0 = 0.5 * (ratio[..., 1:] + ratio[..., :-1])
        r1 = (ratio[..., 1:] - ratio[..., :-1]) / arrays["bin_widths"]
        d_inner_h = jnp.sum(arrays["a0"] * jnp.conj(r0) + arrays["a1"] * jnp.conj(r1))
        h_inner_h = jnp.sum(
            arrays["b0"] * r0[:, :, None] * jnp.conj(r0[:, None, :])
            + arrays["b1"] * (r0[:, :, None] * jnp.conj(r1[:, None, :])
                              + jnp.conj(r0[:, None, :]) * r1[:, :, None]))
        return jnp.real(d_inner_h) - jnp.real(h_inner_h) / 2

    def _inputs(self, parameters):
        """Converted parameter arrays for the JAX function."""
        likelihood = self.likelihood
        for key in ("vG", "a", "A"):
            if key in parameters:
                raise NotImplementedError("the batched likelihood supports general relativity only")
        parameters = {key: np.asarray(value) for key, value in parameters.items()
                      if np.asarray(value).dtype.kind in "fiub"}
        size = max((value.size for value in parameters.values()), default=1)
        parameters = {key: np.broadcast_to(value, (size,)).astype(float)
                      for key, value in parameters.items()}
        parameters.update(likelihood.get_sky_frame_parameters(parameters))
        converted, _ = likelihood.waveform_generator.parameter_conversion(parameters)
        times = np.asarray(converted["geocent_time"], dtype=float)
        gmst = np.array([greenwich_mean_sidereal_time(t) for t in times])
        gmst_rate = (np.array([greenwich_mean_sidereal_time(t + _DAY) for t in times]) - gmst) / _DAY
        q = np.asarray(converted["q"], dtype=float)
        inputs = dict(
            q=np.where(q <= 1, 1 / q, q), lambda_1=converted["LambdaAl2"],
            lambda_2=converted["LambdaBl2"], chi_1=converted["chi1z"], chi_2=converted["chi2z"],
            total_mass=converted["M"], distance=converted["distance"],
            inclination=converted["inclination"], phase=converted["coalescence_angle"],
            mass_1=converted["mass_1"], mass_2=converted["mass_2"], ra=converted["ra"],
            dec=converted["dec"], psi=converted["psi"], geocent_time=times,
            gmst=gmst, gmst_rate=gmst_rate)
        return {key: np.broadcast_to(np.asarray(value, dtype=float), (size,))
                for key, value in inputs.items()}, size

    def log_likelihood_ratio(self, parameters=None):
        """ln L ratio at ``parameters``, a dict of scalars (returns a float)
        or of equal-length arrays (returns an array)."""
        if parameters is None:
            parameters = self.parameters
        scalar = all(np.ndim(value) == 0 for value in parameters.values())
        inputs, size = self._inputs(parameters)
        out = np.empty(size)
        chunk = 1 if size == 1 else self.batch_size
        for start in range(0, size, chunk):
            stop = min(start + chunk, size)
            batch = {key: np.pad(value[start:stop], (0, chunk - (stop - start)), mode="edge")
                     for key, value in inputs.items()}
            out[start:stop] = np.asarray(self._batched(batch, self._arrays))[:stop - start]
        return float(out[0]) if scalar else out

    def noise_log_likelihood(self):
        return self.likelihood.noise_log_likelihood()

    def log_likelihood(self, parameters=None):
        return self.log_likelihood_ratio(parameters) + self.noise_log_likelihood()


def _batched_bilby_model_classes():
    from nessai_bilby.model import BilbyModel

    class BatchedBilbyModel(BilbyModel):
        """nessai model evaluating a :class:`BatchedRelativeBinningLikelihood`
        on whole batches of live points."""

        allow_vectorised = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # batch and single evaluations agree to rounding, not to the
            # 1e-15 nessai's automatic check demands
            self.vectorised_likelihood = True
            self.likelihood_chunksize = getattr(self.bilby_likelihood, "batch_size", None)

        def _theta(self, x):
            theta = dict(self.fixed_parameters)
            for name in self.names:
                theta[name] = np.asarray(x[name], dtype=float)
            return theta

        def log_likelihood(self, x):
            theta = self._theta(x)
            out = np.atleast_1d(self.bilby_log_likelihood_fn(theta))
            return out.reshape(np.shape(x)) if np.ndim(x) else out[0]

    class BatchedBilbyModelLikelihoodConstraint(BatchedBilbyModel):
        """As :class:`BatchedBilbyModel`, with the prior constraints in the
        likelihood."""

        def log_likelihood(self, x):
            theta = self._theta(x)
            allowed = np.atleast_1d(self.bilby_priors.evaluate_constraints(theta))
            out = np.atleast_1d(self.bilby_log_likelihood_fn(theta))
            out = np.where(allowed, out, -np.inf)
            return out.reshape(np.shape(x)) if np.ndim(x) else out[0]

        def log_prior(self, x):
            theta = {n: x[n] for n in self.names}
            return self.bilby_priors.ln_prob(theta, axis=0)

    return BatchedBilbyModel, BatchedBilbyModelLikelihoodConstraint


def __getattr__(name):
    # the nessai classes are built lazily, so nessai stays optional
    if name in ("BatchedBilbyModel", "BatchedBilbyModelLikelihoodConstraint"):
        return dict(zip(("BatchedBilbyModel", "BatchedBilbyModelLikelihoodConstraint"),
                        _batched_bilby_model_classes()))[name]
    if name == "BatchedNessai":
        return _batched_nessai_class()
    raise AttributeError(name)


_BATCHED_NESSAI = None


def _batched_nessai_class():
    global _BATCHED_NESSAI
    if _BATCHED_NESSAI is None:
        from nessai_bilby import plugin

        model, constrained = _batched_bilby_model_classes()

        class BatchedNessai(plugin.Nessai):
            """bilby's nessai sampler, with the likelihood evaluated on whole
            batches of points: ``bilby.run_sampler(likelihood=
            BatchedRelativeBinningLikelihood(...), sampler=BatchedNessai, ...)``.
            """

            def run_sampler(self):
                originals = plugin.BilbyModel, plugin.BilbyModelLikelihoodConstraint
                plugin.BilbyModel, plugin.BilbyModelLikelihoodConstraint = model, constrained
                try:
                    return super().run_sampler()
                finally:
                    plugin.BilbyModel, plugin.BilbyModelLikelihoodConstraint = originals

        _BATCHED_NESSAI = BatchedNessai
    return _BATCHED_NESSAI
