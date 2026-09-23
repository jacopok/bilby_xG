"""Tests for the surrogate / EOB BNS source models (optional dependencies)."""
import numpy as np
import pytest

PARAMETERS = dict(chirp_mass=1.2, mass_ratio=0.9, chi_1=0.01, chi_2=0.0,
                  lambda_1=400.0, lambda_2=900.0, luminosity_distance=40.0,
                  theta_jn=0.35, phase=1.57)
SOURCE_KEYS = ["M", "q", "chi1z", "chi2z", "LambdaAl2", "LambdaBl2", "distance",
               "inclination", "coalescence_angle"]


def test_mlgw_bns_modes():
    pytest.importorskip("mlgw_bns")
    from bilby_xG.source import convert_to_mlgw_bns_parameters, mlgw_bns_individual_modes

    params, added = convert_to_mlgw_bns_parameters(dict(PARAMETERS))
    assert {"M", "q", "LambdaAl2", "coalescence_angle"} <= set(added)
    freqs = np.array([0.0, 5.0, 20.0, 100.0, 1000.0])
    kwargs = {key: params[key] for key in SOURCE_KEYS}
    modes = mlgw_bns_individual_modes(freqs, **kwargs, mode_array=[[2, 2], [3, 3]])
    assert set(modes) == {"2,2", "3,3"}
    for pols in modes.values():
        assert pols["plus"][0] == 0 and np.all(np.abs(pols["plus"][1:]) > 0)
    # frequency_bin_edges takes precedence over the frequency array
    on_edges = mlgw_bns_individual_modes(freqs, **kwargs, mode_array=[[2, 2]],
                                         frequency_bin_edges=freqs[2:])
    np.testing.assert_array_equal(on_edges["2,2"]["cross"], modes["2,2"]["cross"][2:])
    with pytest.raises(ValueError):
        mlgw_bns_individual_modes(freqs, **kwargs, mode_array=[[3, 2]])


def test_teobresums_spa_modes():
    pytest.importorskip("EOBRun_module")
    from bilby_xG.source import (
        convert_to_teobresums_parameters,
        teobresums_spa_individual_modes,
    )

    params, added = convert_to_teobresums_parameters(dict(PARAMETERS))
    assert "use_spins" not in added and params["chi1"] == PARAMETERS["chi_1"]
    kwargs = {key: params[key] for key in SOURCE_KEYS}
    freqs = np.geomspace(20, 1024, 200)
    modes = teobresums_spa_individual_modes(
        freqs, **kwargs, reference_frequency=20.0, minimum_frequency=20.0,
        maximum_frequency=1024.0, mode_array=[[2, 2], [3, 3]],
        frequency_bin_edges=freqs)
    assert set(modes) == {"2,2", "3,3"}
    h22, h33 = (np.abs(modes[k]["plus"]) for k in ("2,2", "3,3"))
    assert np.all(h22 > 0) and np.all(h22 > h33)
    # SPA amplitude of the (2,2) inspiral falls as f^(-7/6)
    slope = np.polyfit(np.log(freqs[:50]), np.log(h22[:50]), 1)[0]
    assert slope == pytest.approx(-7 / 6, abs=0.05)
