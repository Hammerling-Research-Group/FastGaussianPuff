# FastGaussianPuff
*Production code the Fast Implementation of the Gaussian Puff Forward Atmospheric Model*

[![build](https://github.com/Hammerling-Research-Group/FastGaussianPuff/actions/workflows/build.yml/badge.svg)](https://github.com/Hammerling-Research-Group/FastGaussianPuff/actions/workflows/build.yml)
[![Check Link Rot](https://github.com/Hammerling-Research-Group/FastGaussianPuff/actions/workflows/check-link-rot.yaml/badge.svg)](https://github.com/Hammerling-Research-Group/FastGaussianPuff/actions/workflows/check-link-rot.yaml)

This repository contains multiple different implementations of the Gaussian puff atmospheric dispersion model that simulates concentration timeseries given a geometry and emission parameters. The Gaussian puff model simulates a continuous emission as a series of discrete puffs. As long as puffs are emitted often enough and tracked finely enough (more on that later), this model can give reasonable results more quickly than solving an advection-diffusion equation.

Specifically, the code computes the space and time-dependent concentration field

$$ c(x,y,z,t) = \sum_{p \in S_t} c_p(x,y,z,t) $$

where $S_t$ is the set of active puffs at time $t$ and

$$ c_p(x,y,z,t) = \frac{q}{(2\pi)^{3/2}  \sigma_y^2 \sigma_z} \exp{\left(-\frac{(x-ut)^2+y^2}{2\sigma_y^2}\right)} \left[\exp{\left(-\frac{(z-z_0)^2}{2\sigma_z^2}\right)} + \exp{\left(-\frac{(z+z_0)^2}{2\sigma_z^2}\right)} \right]$$

is the concentration field for a single puff. Here, $q$ is the amount of methane in a puff, $u$ is wind speed, $z_0$ is the emission release height, and $\sigma_{y,z}$ are dispersion parameters that control plume spread.

## How this code works
Fast implementations of algorithms to simulate this model live in some C++ code. This has a Python interface and is designed to be used from Python.

Currently, you need four sets of parameters to set up a simulation.
1. Geometry information. This is the spatial domain you're interested in.
2. Emission parameters. These include where your emission is coming from, how long to simulate for, and at what emission rate.
3. Wind data. You need timeseries for both wind speed and wind direction that are regularly spaced in time.
4. Timestep parameters. These parameters affect accuracy of the simulation. Higher wind speeds or rapid changes in wind direction means these parameters need to be smaller to maintain accuracy. Hopefully one day we can set these automatically based on the wind data.

There are descriptions for each parameter in the [class file](GaussianPuff.py) and demos for how to use the code in the `demos/` directory.

### Site geometry
Currently, we care about two use cases. Each of these have smart implementations that are specialized to be fast for each scenario and require different 
1. Regularly gridded rectangular domain. Here, we create a 3D rectangular grid and simulate concentrations at each point. This is useful for visualization, among other things.
2. Sparse points-in-space. This is intended for when you only care about a few specific points in the domain (e.g. simulating concentration timeseries for point-in-space sensors).

### Time-related parameters
There are four time-related parameters to distinguish between:
1. `sim_dt`: This is the main simulation time step. If this parameter is too high it will cause "skipping" in the simulation, like a movie with a bad framerate. What constitutes as "too high" depends on the wind speed.
2. `puff_dt`: This is how frequently puffs are released. While it would seem to be bad to emit puffs infrequently, the right value for this parameter depends on how rapidly the wind direction is changing. If the direction is relatively constant, this parameter can be set higher due to how concentration values are averaged over time for each puff. While not exactly a simulation timestep, this parameter plays a role in accuracy.
3. `obs_dt`: This is not a timestep parameter. Instead, this should be how far apart in time the observations are in the wind data. E.g. `obs_dt=60` means you have exactly one data point every 60 seconds.
4. `output_dt`: This is not a timestep parameter and is not required to be set. This is the resolution in time to output the final concentration timeseries, e.g. `output_dt=60` will provide a concentration timeseries with one data point every minute. By default, this is set to be the same as `obs_dt`. 

There are a few restrictions imposed on the time parameters.
1. They should all be positive.
2. We should have both `puff_dt` > `sim_dt` and `out_dt` > `sim_dt`. Note that while we can have `puff_dt` > `obs_dt`, it is not advised except for specific visualization purposes. This is because data is being output between the creation of individual puffs, so you will see individual puffs traveling.
3. `puff_dt` should be a positive integer multiple of `sim_dt`, i.e. `puff_dt` = `n*sim_dt` for some positive integer n. This prevents the code having to interpolate the concentration values in time, although it is likely that this constraint could be avoided.

To simulate below the resolution of `obs_dt`, wind data is interpolated to resolution `puff_dt` so that each puff may have a separate wind direction and speed.

### Timezones
The module requires the input timestamps to be time zone-aware, and the input time zone string should correspond to the location of the site to be simulated. For example, simulating a site in Colorado could use `time_zone="America/Denver"`. The start/end times can be in any time zone but will get converted to the specified time for simulation (e.g. start/end could be input in UTC and will get converted the America/Denver). For a list of all timezones, use the following:
```python
import zoneinfo
zoneinfo.available_timezones()
```

## Installation instructions
**Currently, this install process will not work on Windows.**

Python 3.9 or higher is required. We highly recommend using a [conda](https://docs.conda.io/en/latest/) environment. You can create the environment with

```shell
$ conda env create -f environment.yml
```

Then, activate the environment with:

```shell
$ conda activate gp
```

The module works with pip. To install, use:
```shell
$ pip install .
```

Alternatively, you can install manually using CMake. You can compile and install everything with:

```shell
$ mkdir build && cd build
$ cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
$ make all install
```
It is advisable to install the library in the conda environment so that the python bindings are available. The environment variable $CONDA_PREFIX is set to the root of the conda environment.

## Danger zone
This section contains details on special parameters that may cause erroneous output and have specific use-cases. When in doubt, re-run the model with default parameters and compare.

`skip_low_wind` (bool), `low_wind_cutoff` (float, [m/s])
- When true, skip_low_wind will cause the simulation to skip emitting puffs when the wind speed is below the low_wind_cutoff. This was added for convenience since the model is unreliable in low wind speeds (< 0.5m/s) and doesn't run if there is a wind speed of 0 m/s. 

`exp_thresh_tolerance` (float. [ppm])
- The tolerance for the thresholding algorithm. The model will skip evaluation of any cell that will produce a result less than this. Note that this is not one-to-one with the smallest concentration you'll see in the output due to how output data is resampled in time. As such, this should be set conservatively. Default: 1e-7

`unsafe` (bool)
- Turning this to true enables unsafe math operations that approximate evaluation of expensive functions and uses a tighter threshold. Expect to see between a 1.5-2x speed up, but the output may vary from what is expected. When in doubt, rerun the model with this off and compare.