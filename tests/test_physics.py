import numpy as np
import pytest

from dt.physics import MU, R_E, Faults, OrbitModel, SatellitePhysics


def test_period_matches_kepler():
    o = OrbitModel(altitude_m=515e3)
    expected = 2 * np.pi * np.sqrt((R_E + 515e3) ** 3 / MU)
    assert o.period_s == pytest.approx(expected)
    assert 94 < o.period_s / 60 < 96


@pytest.mark.parametrize("beta", [0, 20, 40, 60, 75])
def test_simulated_eclipse_matches_analytic(beta):
    o = OrbitModel(beta_deg=beta)
    sim = SatellitePhysics(orbit=o).simulate(n_orbits=3, dt_s=5)
    assert 1 - sim["sunlit"].mean() == pytest.approx(o.eclipse_fraction_analytic(), abs=2e-3)


def test_no_eclipse_above_critical_beta():
    o = OrbitModel(beta_deg=70)   # critical beta at 515 km is ~67.7 deg
    assert o.eclipse_fraction_analytic() == 0.0
    assert SatellitePhysics(orbit=o).simulate(n_orbits=2)["sunlit"].all()


def test_soc_bounded_even_with_faults():
    T = OrbitModel().period_s
    sim = SatellitePhysics().simulate(n_orbits=6, faults=Faults(panel_degradation=(T, 0.9),
                                                                battery_fade=(2 * T, 0.5)))
    assert sim["soc"].min() >= 0.0 and sim["soc"].max() <= 1.0


def test_faults_change_behaviour_only_after_onset():
    sp = SatellitePhysics()
    T = sp.orbit.period_s
    nom = sp.simulate(n_orbits=4)
    bad = sp.simulate(n_orbits=4, faults=Faults(panel_degradation=(2 * T, 0.5)))
    before = nom["t_s"] < 2 * T
    np.testing.assert_allclose(nom["solar_w"][before], bad["solar_w"][before])
    assert bad["solar_w"][~before].sum() < nom["solar_w"][~before].sum()
    assert bad["fault"][~before].all() and not bad["fault"][before].any()


def test_sensor_stuck_freezes_channel():
    sp = SatellitePhysics()
    sim = sp.simulate(n_orbits=2, faults=Faults(sensor_stuck=(1000.0, "temp_k")))
    stuck = sim["temp_k"][sim["t_s"] >= 1000.0]
    assert np.all(stuck == stuck[0])
