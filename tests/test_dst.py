import pytest
import numpy as np
import pandas as pd
from utils import params, grid_params, sensor_params
from FastGaussianPuff import GaussianPuff as GP


@pytest.mark.parametrize(("mode_params"), [grid_params, sensor_params])
@pytest.mark.parametrize(
    ("dst_start", "dst_end", "expected_length"),
    [
        ("2024-03-09 12:00:00-07:00", "2024-03-10 12:00:00-06:00", 23 * 2),
        ("2024-11-02 12:00:00-06:00", "2024-11-03 12:00:00-07:00", 25 * 2),
        ("2024-03-09 12:00:00+00:00", "2024-03-11 11:00:00+00:00", 47 * 2),
        ("2024-11-02 12:00:00+00:00", "2024-11-04 12:00:00+00:00", 48 * 2),
    ],
)
def test_length_over_dst(mode_params, dst_start, dst_end, expected_length):
    test_params = params.copy()

    test_params["simulation_start"] = dst_start
    test_params["simulation_end"] = dst_end
    test_params["wind_speeds"] = [2.0]*expected_length  # every half hour
    test_params["wind_directions"] = [180.0]*expected_length
    test_params["obs_dt"] = 60*30
    test_params["sim_dt"] = 60
    test_params["puff_dt"] = 60

    gp = GP(**test_params, **mode_params)
    gp.simulate()

    length = len(gp.ch4_obs)

    assert(length == expected_length + 1) # TODO fix the off by one error in the interface


@pytest.mark.parametrize(("mode_params"), [grid_params, sensor_params])
def test_different_timezones(mode_params):
    test_params = params.copy()

    test_params["simulation_start"] = "2024-03-09 12:00:00-07:00"
    test_params["simulation_end"] = "2024-03-10 12:00:00-06:00"
    test_params["wind_speeds"] = [2.0] * 23 * 2

    theta = np.linspace(0, 2 * np.pi, 23 * 2)
    wind_dir = np.cos(theta) * 180 + 180
    test_params["wind_directions"] = wind_dir
    test_params["obs_dt"] = 60 * 30
    test_params["sim_dt"] = 60
    test_params["puff_dt"] = 60

    mtn_gp = GP(**test_params, **mode_params)
    mtn_gp.simulate()
    mtn_ch4 = mtn_gp.ch4_obs

    test_params["simulation_start"] = "2024-03-09 19:00:00+00:00"
    test_params["simulation_end"] = "2024-03-10 18:00:00+00:00"
    utc_gp = GP(**test_params, **mode_params)
    utc_gp.simulate()
    utc_ch4 = utc_gp.ch4_obs

    assert np.allclose(mtn_ch4, utc_ch4, atol=1e-6)
