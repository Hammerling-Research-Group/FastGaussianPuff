from FastGaussianPuff import GaussianPuff
from FastGaussianPuff import GaussianPlume


def test_plume_matches_long_time_puff():
    import numpy as np
    import pandas as pd

    simulation_start = pd.Timestamp("2024-01-01T12:00:00Z")
    simulation_end = pd.Timestamp("2024-01-01T12:30:00Z")
    time_zone = "America/Denver"
    source_coordinates = np.array([[10.0, 22.0, 2.5]])
    emission_rate = 1.0
    wind_speed = 3.5
    wind_direction = 26.0
    grid_coordinates = np.array([0, 0, 0, 50, 50, 10])
    nx = ny = nz = 25
    x = np.linspace(grid_coordinates[0], grid_coordinates[3], nx)
    y = np.linspace(grid_coordinates[1], grid_coordinates[4], ny)
    z = np.linspace(grid_coordinates[2], grid_coordinates[5], nz)
    X, Y, Z = np.meshgrid(x, y, z)

    plume = GaussianPlume(
        simulation_timestamp=simulation_start,
        time_zone=time_zone,
        source_coordinates=source_coordinates,
        emission_rate=emission_rate,
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        X=X,
        Y=Y,
        Z=Z,
    )

    n_mins = simulation_end.minute - simulation_start.minute
    wind_speed_puff = [wind_speed] * n_mins
    wind_dir_puff = [wind_speed] * n_mins
    # Create GaussianPuff instance and simulate
    puff = GaussianPuff(
        simulation_start=simulation_start,
        simulation_end=simulation_end,
        time_zone=time_zone,
        source_coordinates=source_coordinates,
        emission_rates=[emission_rate],
        wind_speeds=wind_speed_puff,
        wind_directions=wind_dir_puff,
        grid_coordinates=grid_coordinates,
        nx=nx,
        ny=ny,
        nz=nz,
        obs_dt=60,
        sim_dt=1,
        puff_dt=1,
    )

    c_puff = puff.simulate()
    c_plume = plume.simulate()

    c_puff_final = c_puff[-1, :]
    np.allclose(c_plume.flatten(), c_puff_final, rtol=1e-5, atol=1e-8)
