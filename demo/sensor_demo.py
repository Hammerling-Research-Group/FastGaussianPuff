import sys
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

code_dir = '../'
sys.path.insert(0, code_dir)

from utilities import wind_synthesizer
from FastGaussianPuff import GaussianPuff as GP

# set simulation parameters
# IMPORTANT: obs_dt must be a positive integer multiple of sim_dt, and both sim_dt and puff_dt must be integers
obs_dt, sim_dt, puff_dt = 60, 1, 1

# start and end times at minute resolution. Needs to be in the local timezone of where we're simulating
# e.g. if we're simulating a site in England, it needs to be in UTC.
# if we're simulating a site in Colorado, it should be in MST/MDT
start = pd.to_datetime('2022-01-01 12:00:00')
end = pd.to_datetime('2022-01-01 13:00:00')

# fabricated wind data
fake_times = np.linspace(0,10,61)
wind_speeds = [3]*61
wind_directions = 120*np.abs(np.cos(fake_times))
wind_directions[30:60] -= 40*np.abs(np.sin(6*fake_times[30:60]))



# emission source
source_coordinates = [[488163.338444176, 4493892.53205817, 2.0]] # format is [[x0,y0,z0]] in [m]. needs to be nested list for compatability with multi source (coming soon)
emission_rate = [3.5] # emission rate for the single source above, [kg/hr]

# sensors on the site. it is assumed that these encase the source coordinates.
sensor_coordinates = [[488164.98285821447, 4493931.649887275, 2.4],
    [488198.08502694493, 4493932.618594243, 2.4],
    [488226.9012860443, 4493887.916890612, 2.4],
    [488204.9825329503, 4493858.769131294, 2.4],
    [488172.4989330686, 4493858.565324413, 2.4],
    [488136.3904409793, 4493861.530987777, 2.4],
    [488106.145508258, 4493896.167438727, 2.4],
    [488133.15254321764, 4493932.355431944, 2.4]]

sp = GP(obs_dt=obs_dt, sim_dt=sim_dt, puff_dt=puff_dt,
                 simulation_start=start, simulation_end=end,
                 source_coordinates=source_coordinates, emission_rates=emission_rate,
                 wind_speeds=wind_speeds, wind_directions=wind_directions,
                 using_sensors=True, sensor_coordinates=sensor_coordinates, 
                 quiet=True # change to false for progress information
)

sp.simulate()

#%% plotting
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


fig.savefig("demo_sensors.png", format="png", dpi=500, bbox_inches="tight")
