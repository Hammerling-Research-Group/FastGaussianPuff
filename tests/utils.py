import pandas as pd
import numpy as np

fake_times = np.linspace(0,10,61)
wind_speeds = [3]*61
wind_directions = 120*np.abs(np.cos(fake_times))
wind_directions[30:60] -= 40*np.abs(np.sin(6*fake_times[30:60]))

wind_speeds = np.array(wind_speeds)
wind_directions = np.array(wind_directions)

params = {
    "obs_dt": 60,
    "sim_dt": 1,
    "puff_dt": 5,
    "simulation_start": pd.to_datetime("2022-01-01 12:00:00").tz_localize("UTC"),
    "simulation_end": pd.to_datetime("2022-01-01 13:00:00").tz_localize("UTC"),
    "emission_rates": np.array([3]),
    "source_coordinates": np.array([[20, 10, 5]]),
    "wind_speeds": wind_speeds,
    "wind_directions": wind_directions,
    "time_zone": "America/Denver",
}

grid_params = {
    "nx": 10,
    "ny": 10,
    "nz": 10,
    "grid_coordinates": np.array([0, 0, 0, 50, 50, 10]),
}

sensor_params = {"sensor_coordinates" : np.array([[5, 5, 5], [16, 19, 4], [47, 4, 1]]), "using_sensors":True}
