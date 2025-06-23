import pytest
import numpy as np
import pandas as pd
from utils import params, grid_params, sensor_params
from FastGaussianPuff import GaussianPuff as GP


def test_init():
    gp = GP(**params, **grid_params)
    sp = GP(**params, **sensor_params)

def test_list_init():
    ws = [3]*61
    wd = [200]*61

    # this should still work if we pass lists instead of numpy arrays
    params["wind_speeds"] = ws
    params["wind_directions"] = wd
    params["emission_rates"] = [3]
    grid_params["grid_coordinates"] = [0, 0, 0, 50, 50, 10]
    sensor_params["sensor_coordinates"] = [[5, 5, 5], [16, 19, 4], [47, 4, 1]]
    params["source_coordinates"] = [[25, 25, 5]]

    gp = GP(**params, **grid_params)
    sp = GP(**params, **sensor_params)

def test_bad_list_init():
    ws = [3]*61

    test_params = params.copy()

    test_params["wind_speeds"] = pd.DataFrame(ws)

    with pytest.raises(TypeError):
        gp = GP(**test_params, **grid_params)
    with pytest.raises(TypeError):
        sp = GP(**test_params, **sensor_params)


def test_source_coordinate_formats():
    test_params = params.copy()

    # test with a single source coordinate
    test_params["source_coordinates"] = [[25, 25, 5]]
    gp = GP(**test_params, **grid_params)
    gp.simulate()
    ans = np.linalg.norm(gp.ch4_obs)
    assert ans > 0
    # print(np.linalg.norm(gp.ch4_obs))

    # test with a single source coordinate as a list
    test_params["source_coordinates"] = [25, 25, 5]
    gp = GP(**test_params, **grid_params)
    gp.simulate()
    assert np.linalg.norm(gp.ch4_obs) == ans

    # test with multiple source coordinates
    test_params["source_coordinates"] = [[25, 25, 5], [30, 30, 5]]
    with pytest.raises(NotImplementedError):
        gp = GP(**test_params, **grid_params)


def test_naive_tz():
    test_params = params.copy()
    test_params["simulation_start"] = pd.to_datetime("2022-01-01 12:00:00")
    with pytest.raises(ValueError):
        gp = GP(**test_params, **grid_params)

    test_params = params.copy()
    test_params["simulation_end"] = pd.to_datetime("2022-01-01 13:00:00")
    with pytest.raises(ValueError):
        gp = GP(**test_params, **grid_params)

    test_params = params.copy()
    test_params["time_zone"] = "bad_tz"
    with pytest.raises(ValueError):
        gp = GP(**test_params, **grid_params)
