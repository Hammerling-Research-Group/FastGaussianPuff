import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from FastGaussianPuff import GaussianPuff as GP

# set simulation parameters
# IMPORTANT: obs_dt must be a positive integer multiple of sim_dt
obs_dt, sim_dt, puff_dt = 60, 1, 1

# start and end times need to be timezone-aware.
# start/end are the times you want to simulate between. time_zone is the location you want to simulate in.
# you can input start and end times in any timezone and they'll get converted to the time_zone you specify.
# (e.g. input start/end in UTC and set time_zone to "America/Denver" to simulate in mountain time)
start = pd.to_datetime("2022-01-01 12:00:00-06:00")
end = pd.to_datetime("2022-01-01 13:00:00-06:00")
time_zone = "America/Denver"  # alternative: "US/Mountain"

# fabricated wind data
fake_times = np.linspace(0,10,61)
wind_speeds = [3]*61
wind_directions = 120*np.abs(np.cos(fake_times))
wind_directions[30:60] -= 40*np.abs(np.sin(6*fake_times[30:60]))

wind_speeds = np.array(wind_speeds)
wind_directions = np.array(wind_directions)


# emission source
source_coordinates = np.array([[0.0, 0.0, 2.5]]) # format is [[x0,y0,z0]] in [m]. needs to be nested list for compatibility with multi-source emissions
emission_rate = np.array([3.5]) # emission rate for the single source above, [kg/hr]

# sensors on the site. it is assumed that these encase the source coordinates.
theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
r = 30 + 15*np.random.rand(8)
sensor_coordinates = [[r[i]*np.cos(theta[i]), r[i]*np.sin(theta[i]), 1 + 5*np.random.rand()] for i in range(8)]


sp = GP(
    obs_dt=obs_dt,
    sim_dt=sim_dt,
    puff_dt=puff_dt,
    simulation_start=start,
    simulation_end=end,
    source_coordinates=source_coordinates,
    emission_rates=emission_rate,
    wind_speeds=wind_speeds,
    wind_directions=wind_directions,
    using_sensors=True,
    sensor_coordinates=sensor_coordinates,
    time_zone=time_zone,
)

print("STARTING SIMULATION")
sp.simulate()
print("SIMULATION FINISHED")

print("MAKING PLOTS")
# %% plotting
t, n_sensors = np.shape(sp.ch4_obs) # (time, sensors)
sensor_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

fig, ax = plt.subplots(2, 4, figsize=(10,10))
m = sp.ch4_obs.max()
fig.supxlabel("Time from emission start (minutes)")
fig.supylabel("Methane concentration (ppm)")

for i in range(0,n_sensors):

    if i < 4:
        row = 0
        col = i
    else:
        row = 1
        col = i - 4

    times = np.arange(0, t)
    
    sensor_ch4 = sp.ch4_obs[:,i]

    ax[row][col].plot(times, sensor_ch4)
    ax[row][col].set_ylim(-1,m+2)
    ax[row][col].set_title(sensor_names[i])


print("CHECK FILE demo_sensors.png")
fig.savefig("demo_sensors.png", format="png", dpi=500, bbox_inches="tight")
