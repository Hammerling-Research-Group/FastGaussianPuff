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
    wd = [200]*61

    # this should still work if we pass lists instead of numpy arrays
    params["wind_speeds"] = pd.DataFrame(ws)

    with pytest.raises(TypeError):
        gp = GP(**params, **grid_params)
    with pytest.raises(TypeError):
        sp = GP(**params, **sensor_params)