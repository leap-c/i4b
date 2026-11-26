"""A step's energy is the time average of the heat pump's power over the step."""

import numpy as np

from i4b.data.buildings.sfh_1919_1948 import sfh_1919_1948_0_soc
from i4b.models.model_buildings import Building
from i4b.models.model_hvac import Heatpump_AW
from i4b.simulator import Model_simulator


def test_step_energy_is_a_time_average():
    hp = Heatpump_AW(mdot_HP=0.25)
    sim = Model_simulator(hp, Building(params=sfh_1919_1948_0_soc, mdot_hp=0.25, method="4R3C"), 900)
    state = {"T_room": 20.0, "T_wall": 15.0, "T_hp_ret": 25.0}
    p = {"T_amb": 0.0, "Qdot_gains": 0.0}
    res = sim.get_next_state(state, 55.0, p)  # a jump in supply temperature: return water heats up fast

    # Reference: one-second steps, power averaged over time.
    T_ret, x = [], dict(state)
    for _ in range(900):
        T_ret.append(x["T_hp_ret"])
        x = Model_simulator(hp, sim.bldg_model, 1).get_next_state(x, 55.0, p)["state"]
    q_th = np.mean([max(0.25 * 4181 * (55.0 - t), 0.0) for t in T_ret])
    e_ref = q_th / hp.COP(55.0, 0.0) * 900 / 3600  # Wh
    assert abs(res["cost"]["E_el"] - e_ref) <= 0.005 * e_ref
