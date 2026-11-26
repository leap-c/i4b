"""The CasADi building models match their NumPy counterparts."""

import casadi as cas
import numpy as np
import pytest

from i4b.data.buildings.sfh_2016_now import sfh_2016_now_0_soc
from i4b.models.model_buildings import Building

N_INPUTS = {"2R2C": 2, "4R3C": 2, "5R4C": 2, "7R5C": 4}


@pytest.mark.parametrize("method", list(N_INPUTS))
def test_casadi_rhs_matches_numpy(method):
    building = Building(params=sfh_2016_now_0_soc, mdot_hp=0.25, method=method)
    nx, npar = len(building.state_keys), N_INPUTS[method]
    x, u, p = cas.SX.sym("x", nx), cas.SX.sym("u"), cas.SX.sym("p", npar)
    rhs = cas.Function("rhs", [x, u, p], [building.calc_casadi(x, u, p)])
    numpy_rhs = getattr(building, f"calc_{method.lower()}")

    rng = np.random.default_rng(0)
    for _ in range(5):
        xv = rng.uniform(15.0, 40.0, nx)
        uv = rng.uniform(20.0, 55.0)
        pv = np.r_[rng.uniform(-10.0, 15.0), rng.uniform(0.0, 3000.0, npar - 1)]
        expected = np.asarray(numpy_rhs(0.0, xv, uv, pv), float).reshape(-1)
        actual = np.asarray(rhs(xv, uv, pv)).reshape(-1)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-14)
