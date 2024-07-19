# Parser Instructions
This parser is designed to make some common use-cases easier to run. The functionality is necessarily limited and thus the parser will not support all cases and may have undocumented behavior outside of its tested formats. 

# How to use
The parser reads an input file that points to the data and provides parameters, then constructs and runs the simulation. Some examples of these files are `dlq.in` and `multi.in` within this directory. 
## Data sources
There are four data files necessary. It is recommended to give the file paths relative to the input file.
1. Source file: this has coordinates for the relevant sources. Coordinates can be given in lat/lon or in cartesian coordinates. If provided in lat/lon, coordinates will be converted to UTM.
2. Sensor file: contains coordinates of locations where simulation results should be computed. Coordinates can be given in the same way as the source file. The column headers for coordinates are expected to be the same in the source and sensor file.
3. Wind file: wind timeseries. This file has a number of requirements. Namely,
    - Requires a `timestamp` column with timezone-aware datetimes. 
    - The temporal spacing of the wind data should be uniform and equal to the parameter `wind_dt`, discussed below. 
    - Expects two columns, `wind_speed` and `wind_dir`
    - Columns should contain no gaps or NANs. 
4. Experiments file: this dictates what simulations will be run. It requires a `start_time` and an `end_time` (timezone-aware datetimes, like the wind file), the name(s) of the emitting source(s), and their respective emission rates in kg/hr. This file format is very particular- pay attention to quotations and brackets in the file. Examples are in the `in/` directory.

Currently, most column names are hard-coded and thus any user-created input files need to follow the same format. For the source and sensor files specifically, the names of the columns can be set using `coordinate_columns` in the input file. See examples for details.

## Parameters
There are two simulation parameters, an input-dependent parameter, and an output parameter.
1. `sim_dt`: the simulation timestep, in seconds. This determines the highest resolution of data available in the simulation. Recommended to be set at 1 for most cases, but lower may be necessary to resolve scenarios with high wind speeds.
2. `puff_dt`: how frequently puffs are created, in seconds. Recommended to be set between 1 and 4 for most cases. Must be an integer multiple of `sim_dt`. 
3. `wind_dt`: the temporal spacing between wind samples. This is determined by your wind data.
4. `out_dt`: the resolution you want concentration data output as. Defaults to `wind_dt` if not provided. This can't be smaller than `sim_dt`. If a value larger than `sim_dt` is provided, data at the simulation resolution is downsampled. 

## Output format
Output filenames follow the convention `emissionStartTime_exp_n.csv` where `n` is a number corresponding to the row of the experiments file the emission parameters came from. The number is appended to prevent identical filenames in cases where multiple experiments are run with the same emission start time.

The output file contains one column for each set of sensor coordinates provided, plus a `timestamp` column. Be aware that daylight savings is not currently handled correctly, so an experiment ran over a daylight savings time change will likely fail or be incorrect.