"""TEOBResumS in mlgw_bns' conventions (bilby_xG.teobresums_mlgw) against
the mlgw_bns sources at the same parameters, nothing optimised: they agree
to about the surrogate's accuracy (needs mlgw_bns 2.1 and EOBRun_module)."""
import numpy as np
import pytest

pytest.importorskip("EOBRun_module")

from bilby_xG.source import (MLGW_BNS_MODES, mlgw_bns_individual_modes,  # noqa: E402
                             mlgw_bns_precessing_individual_modes)
from bilby_xG.teobresums_mlgw import (teobresums_mlgw_individual_modes,  # noqa: E402
                                      teobresums_mlgw_precessing_individual_modes)

#: the GW170817-like injection of xg_pe's teob_injections (2000 model)
M = 2.7357
COMMON = dict(M=M, q=1.27 / 1.46, LambdaAl2=234.9, LambdaBl2=588.7, distance=40.0)
FREQS = np.geomspace(25.0, 2000.0, 3000)
#: a chunk of the data grid at its bottom (df = 1 / 25000 s), where the
#: phase advances ~6 rad per sample
LOW = 3.0 + np.arange(2 ** 16) / 25000.0
WEIGHT = np.gradient(FREQS) / FREQS ** (7 / 3)
ARGS = dict(minimum_frequency=3.0, mode_array=MLGW_BNS_MODES)


def mismatch(a, b):
    """1 - Re<a|b> / |a||b|: nothing optimised."""
    norm = np.sqrt(np.sum(abs(a) ** 2 * WEIGHT) * np.sum(abs(b) ** 2 * WEIGHT))
    return 1 - np.sum(np.conj(a) * b * WEIGHT).real / norm


def total(modes, key):
    return sum(mode[key] for mode in modes.values())


def test_aligned_matches_the_surrogate():
    parameters = COMMON | dict(chi1z=0.0029, chi2z=0.0014, inclination=0.3491,
                               coalescence_angle=1.57)
    teob = teobresums_mlgw_individual_modes(FREQS, **parameters, **ARGS)
    mlgw = mlgw_bns_individual_modes(FREQS, **parameters, **ARGS)
    assert sorted(teob) == sorted(mlgw)
    for key in ("plus", "cross"):
        assert mismatch(total(teob, key), total(mlgw, key)) < 1e-6
    # each mode's time-frequency map, the surrogate's (a few us typically),
    # well below the mode's merger (~600 m Hz here)
    for mode in teob:
        band = FREQS < 300 * int(mode.split(",")[1])
        t_teob, t_mlgw = (x[mode]["time_to_merger"][band] for x in (teob, mlgw))
        relative = abs(t_teob - t_mlgw) / t_mlgw
        # a few isolated points (47.3 Hz in every mode) differ by up to ~2e-3
        assert np.median(relative) < 1e-4 and np.max(relative) < 1e-2


def test_aligned_on_the_data_grid():
    """A chunk of the data grid at its bottom, below the surrogate's trained
    band (its PN extension), and two overlapping chunks, which agree where
    they overlap (the injection evaluates chunk by chunk)."""
    parameters = COMMON | dict(chi1z=0.0029, chi2z=0.0014, inclination=0.3491,
                               coalescence_angle=1.57)
    teob = teobresums_mlgw_individual_modes(LOW, **parameters, **ARGS)
    mlgw = mlgw_bns_individual_modes(LOW, **parameters, **ARGS)
    a, b = total(teob, "plus"), total(mlgw, "plus")
    assert 1 - np.sum(np.conj(a) * b).real / np.sqrt(np.sum(abs(a) ** 2) * np.sum(abs(b) ** 2)) < 1e-4
    t_teob, t_mlgw = teob["2,2"]["time_to_merger"], mlgw["2,2"]["time_to_merger"]
    assert np.median(abs(t_teob / t_mlgw - 1)) < 1e-6
    assert np.max(abs(t_teob / t_mlgw - 1)) < 1e-4
    upper = teobresums_mlgw_individual_modes(LOW[len(LOW) // 2:], **parameters, **ARGS)
    for mode in teob:
        np.testing.assert_allclose(upper[mode]["plus"], teob[mode]["plus"][len(LOW) // 2:],
                                   rtol=1e-8, atol=1e-8 * np.max(abs(teob[mode]["plus"])))


#: the other teob_injections binaries (source frame m1, m2, Lambda1, Lambda2
#: on SLy) with their models; from a 1.36 Hz start, TEOB's spin evolution
#: used to stop after one step below ~2 Hz at these masses (fork 163eabfe)
LOW_MASS = {
    4000: (1.1022, 0.9588, 1375.5, 3000.0),
    8000: (0.9676, 0.8417, 2854.6, 6000.0),
    12000: (0.8944, 0.778, 4358.5, 9000.0),
}


@pytest.mark.parametrize("model", [2000, *LOW_MASS])
def test_precessing_matches_the_surrogate(model):
    common = COMMON
    if model in LOW_MASS:
        m1, m2, l1, l2 = LOW_MASS[model]
        # detector frame at 40 Mpc
        common = dict(M=(m1 + m2) * 1.00905, q=m2 / m1, LambdaAl2=l1, LambdaBl2=l2,
                      distance=40.0)
    parameters = common | dict(a_1=0.05, a_2=0.0014, tilt_1=np.pi / 2, tilt_2=0.0,
                               phi_12=0.0, phi_jl=0.0, theta_jn=0.3491, phase=1.57)
    args = ARGS | dict(reference_frequency=20.0,
                       mlgw_bns_model=f"default_hom_lambda{model}")
    teob = teobresums_mlgw_precessing_individual_modes(FREQS, **parameters, **args)
    mlgw = mlgw_bns_precessing_individual_modes(FREQS, **parameters, **args)
    assert sorted(teob) == sorted(mlgw)
    # 2e-7, 1.4e-8, 1.3e-7, 1.3e-6: the 12000 model is ~7x less accurate
    # than 8000 aligned too (1.8e-7 against 2.7e-8)
    tolerance = 3e-6 if model == 12000 else 1e-6
    for key in ("plus", "cross"):
        assert mismatch(total(teob, key), total(mlgw, key)) < tolerance
    assert np.all(teob["2,2"]["time_to_merger"][FREQS < 1000] > 0)
