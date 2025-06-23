import datetime
from math import floor
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

from FastGaussianPuff import CGaussianPuff as fGP

class GaussianPuff:

    def __init__(
        self,
        obs_dt,
        sim_dt,
        puff_dt,
        simulation_start,
        simulation_end,
        time_zone,
        source_coordinates,
        emission_rates,
        wind_speeds,
        wind_directions,
        output_dt=None,
        using_sensors=False,
        sensor_coordinates=None,
        grid_coordinates=None,
        nx=None,
        ny=None,
        nz=None,
        puff_duration=1200,
        skip_low_wind=False,
        low_wind_cutoff=-1,
        exp_threshold_tolerance=None,
        conversion_factor=1e6 * 1.524,
        unsafe=False,
        quiet=True,
    ):
        """
        Inputs:
            obs_dt [s] (scalar, double):
                time interval (dt) for the observations
                NOTE: must be larger than sim_dt. This should be the resolution of the wind data.
            sim_dt [s] (scalar, double):
                time interval (dt) for the simulation results
            puff_dt [s] (scalar, double):
                time interval (dt) between two successive puffs' creation
                NOTE: must also be a positive integer multiple of sim_dt, e.g. puff_dt = n*sim_dt for integer n > 0
            output_dt [s] (scalar, double):
                resolution to resample the concentration to at the end of the sismulation. By default,
                resamples to the resolution of the wind observations obs_dt.
            simulation_start, simulation_end (pd.DateTime values)
                start and end times for the emission to be simulated. Must be timezone-aware.
            time_zone (timezone string):
                timezone to use for the simulation. This should be a string that can be parsed by
                the zoneinfo module, e.g. "America/New_York".
            source_coordinates (array, size=(n_sources, 3)) [m]:
                holds source coordinate (x0,y0,z0) in meters for each source.
            emission_rates: (array, length=n_sources) [kg/hr]:
                rate that each source is emitting at in kg/hr.
            wind_speeds [m/s] (list of floats):
                wind speed at each time stamp, in obs_dt resolution
            wind_directions [degree] (list of floats):
                wind direction at each time stamp, in obs_dt resolution.
                follows the conventional definition:
                0 -> wind blowing from North, 90 -> E, 180 -> S, 270 -> W
            using_sensors (boolean):
                If True, ignores grid-related input parameters and only simulates at sensor coordinates.
                True inputs:
                    - sensor_coordinates
                False inputs:
                    - grid_coordinates,
                    - nx, ny, nz
            sensor_coordinates: (array, size=(n_sensors, 3)) [m]
                coordinates of the sensors in (x,y,z) format.
            grid_coordinates: (array, length=6) [m]
                holds the coordinates for the corners of the rectangular grid to be created.
                format is grid_coordinates=[min_x, min_y, min_z, max_x, max_y, max_z]
            nx, ny, ny (scalar, int):
                Number of points for the grid the x, y, and z directions
            puff_duration (double) [seconds] :
                how many seconds a puff can 'live'; we assume a puff will fade away after a certain time.
                Depending on the grid size wind speed, this parameter will never come into play as the simulation
                for the puff stops when the plume has moved far away. In low wind speeds, however, this cutoff will
                halt the simulation of a puff early. This may be desirable as exceedingly long (and likely unphysical)
                plume departure times can be computed for wind speeds << 1 m/s
            skip_low_wind (boolean), low_wind_cutoff [m/s] (float):
                if True, the simulation will skip any time step where the wind speed is below low_wind_cutoff.
                This is useful to avoid zero-values or situations where the wind is so slow that it'd create unreasonable predictions.
                Default is False.
            exp_threshold_tolerance (scalar, float):
                the tolerance used to threshold the exponentials when evaluating the Gaussian equation.
                If, for example, exp_tol = 1e-9, the concentration at a single point for an individual time step
                will have error less than 1e-9. Upsampling to different time resolutions may introduce extra error.
                Default is 1e-7, which passess all safe-mode tests with less than 0.1% error.
            conversion_factor (scalar, float):
                convert from kg/m^3 to ppm, this factor is for ch4 only
            unsafe (boolean):
                if True, will use unsafe evaluations for some operations. This mode is faster but introduces some
                error. If you're unsure about results, set to False and compare error between the two methods.
            quiet (boolean):
               if True, outputs extra information about the simulation and its progress.
        """

        self.obs_dt = obs_dt 
        self.sim_dt = sim_dt 
        self.puff_dt = puff_dt
        if output_dt is None:
            self.out_dt = self.obs_dt
        else:
            self.out_dt = output_dt

        self._check_timestep_parameters()

        self.sim_start = self._ensure_utc(simulation_start)
        self.sim_end = self._ensure_utc(simulation_end)

        try:
            time_zone = ZoneInfo(time_zone)
        except ZoneInfoNotFoundError:
            raise ValueError(f"Invalid timezone: {time_zone}")

        utc_total_time_series = pd.date_range(
            start=self.sim_start, end=self.sim_end, freq=f"{puff_dt}s", tz="UTC"
        )
        local_ts = utc_total_time_series.tz_convert(time_zone)
        hours_arr = local_ts.hour.values
        n_puffs = len(hours_arr)

        self.quiet = quiet

        # allow unsafe mode to have coarser thresholding
        if exp_threshold_tolerance is None:
            if unsafe:
                self.exp_threshold_tolerance = 1e-5
            else: 
                self.exp_threshold_tolerance = 1e-7
        else:
            self.exp_threshold_tolerance = exp_threshold_tolerance

        if(skip_low_wind):
            if(low_wind_cutoff <= 0):
                raise ValueError("[FastGaussianPuff] low wind cutoff must be greater than 0")
            self.skip_low_wind = True
            self.low_wind_cutoff = low_wind_cutoff

        ns = (self.sim_end-self.sim_start).total_seconds()
        self.n_obs = floor(ns/obs_dt) + 1 # number of observed data points we have

        arrays = self._check_array_dtypes(wind_speeds, wind_directions, source_coordinates, 
                                     emission_rates, grid_coordinates, sensor_coordinates)
        wind_speeds, wind_directions, source_coordinates, emission_rates, grid_coordinates, sensor_coordinates = arrays

        self._check_wind_data(wind_speeds, skip_low_wind)

        # resample the wind data from obs_dt to the simulation resolution sim_dt
        self._interpolate_wind_data(wind_speeds, wind_directions, puff_dt, self.sim_start, self.sim_end)

        # save timeseries of simulation resolution so we can resample back to observation later
        self.time_stamps_sim = pd.date_range(self.sim_start, self.sim_end, freq=str(self.sim_dt)+"s")
        self.n_sim = len(self.time_stamps_sim) # number of simulation time steps

        source_coordinates = self._parse_source_coords(source_coordinates)

        if puff_duration == None:
            puff_duration = self.n_sim # ensures we don't overflow time index

        # creates grid
        if not using_sensors:
            self.using_sensors = False

            self.nx = nx
            self.ny = ny
            self.nz = nz
            self.N_points = self.nx*self.ny*self.nz

            x_min = grid_coordinates[0]
            y_min = grid_coordinates[1]
            z_min = grid_coordinates[2]
            x_max = grid_coordinates[3]
            y_max = grid_coordinates[4]
            z_max = grid_coordinates[5]

            x, y, z = np.linspace(x_min, x_max, self.nx), np.linspace(y_min, y_max, self.ny), np.linspace(z_min, z_max, self.nz)

            self.X, self.Y, self.Z = np.meshgrid(x, y, z) # x-y-z grid across site in utm
            self.grid_dims = np.shape(self.X)

            # work with the flattened grids
            self.X = self.X.ravel()
            self.Y = self.Y.ravel()
            self.Z = self.Z.ravel()

            # constructor for the c code
            spatial_grid = (self.X, self.Y, self.Z, self.nx, self.ny, self.nz)
            dts = (sim_dt, puff_dt, puff_duration)
            wind = (self.wind_speeds_sim, self.wind_directions_sim)
            self.GPC = fGP.GridGaussianPuff(
                *spatial_grid,
                *dts,
                n_puffs,
                hours_arr,
                *wind,
                source_coordinates,
                emission_rates,
                conversion_factor,
                self.exp_threshold_tolerance,
                skip_low_wind,
                low_wind_cutoff,
                unsafe,
                quiet,
            )
        else:
            self.using_sensors = True
            self.N_points = len(sensor_coordinates)

            self.X, self.Y, self.Z = [], [], []
            for sensor in sensor_coordinates:
                self.X.append(sensor[0])
                self.Y.append(sensor[1])
                self.Z.append(sensor[2])

            spatial_grid = (self.X, self.Y, self.Z, self.N_points)
            dts = (sim_dt, puff_dt, puff_duration)
            wind = (self.wind_speeds_sim, self.wind_directions_sim)
            self.GPC = fGP.SensorGaussianPuff(
                *spatial_grid,
                *dts,
                n_puffs,
                hours_arr,
                *wind,
                source_coordinates,
                emission_rates,
                conversion_factor,
                self.exp_threshold_tolerance,
                skip_low_wind,
                low_wind_cutoff,
                unsafe,
                quiet,
            )

        # initialize the final simulated concentration array
        self.ch4_sim = np.zeros((self.n_sim, self.N_points)) # simulation in sim_dt resolution, flattened

    def _parse_source_coords(self, source_coordinates):
        size = np.shape(source_coordinates)
        if len(size) == 1:
            if size[0] == 3:
                source_coordinates = np.array(
                    [source_coordinates]
                )  # now a nested array- C++ code expects this format
            else:
                print(
                    "[fGP] Error: source_coordinates must be a 3-element array, e.g. [x0, y0, z0]."
                )
                exit(-1)
        else:
            if size[0] == 1 and size[1] == 3:
                return source_coordinates
            elif size[0] > 1 and size[1] == 3:
                raise (
                    NotImplementedError(
                        "[fGP] Error: Multi-source currently isn't implemented. Only provide coordinates for a single source, e.g. [x0, y0, z0] or [[x0, y0, z0]]."
                    )
                )

        return source_coordinates

    def _check_timestep_parameters(self):

        # rationale: negative time seems bad. maybe ask a physicist.
        if self.sim_dt <= 0:
            print("[fGP] Error: sim_dt must be a positive value")
            exit(-1)

        # rationale: breaking this would mean multiple puffs are emitted in between simulation time steps. this constraint
        # decouples sim_dt and puff_dt so that sim_dt can be scaled according to wind speed and puff_dt can be scaled
        # according to the change in wind direction over time to guarantee accuracy of the simulation (i.e. no skipping)
        if self.puff_dt < self.sim_dt:
            print("[fGP] Error: puff_dt must be greater than or equal to sim_dt")
            exit(-1)

        # rationale: concentration arrays are build at sim_dt resolution. puff_dt = n*sim_dt for an integer n > 0
        # ensure that puffs are emitted on simulation time steps and prevents the need for a weird interpolation/rounding.
        # this constaint could likely be avoided, but it isn't that strong and makes the code easier.
        eps = 1e-5
        ratio = self.puff_dt/self.sim_dt
        if abs(ratio - round(ratio)) > eps:
            print("[fGP] Error: puff_dt needs to be a positive integer multiple of sim_dt")
            exit(-1)

        # rationale: we don't have simulation data at a resolution less than sim_dt, so you'll have blank
        # concentration fields if this condition isn't met
        if self.out_dt < self.sim_dt:
            print("[fGP] Error: output_dt must be greater than or equal to sim_dt")
            exit(-1)

    def _check_array_dtypes(self, wind_speeds, wind_directions, source_coordinates, emission_rates, grid_coordinates, sensor_coordinates):
        variables = {
            "wind_speeds": wind_speeds,
            "wind_directions": wind_directions,
            "source_coordinates": source_coordinates,
            "emission_rates": emission_rates,
            "grid_coordinates": grid_coordinates,
            "sensor_coordinates": sensor_coordinates,
        }

        casted_arrays = {}

        for name, var in variables.items():
            if var is not None:
                try:
                    casted_arrays[name] = np.asarray(var, dtype=float)
                except Exception as e:
                    raise ValueError(f"[fGP] Error: Failed to cast '{name}' to a NumPy float array: {e}. Try using an array dtype.")
            else:
                casted_arrays[name] = None

        return (casted_arrays["wind_speeds"], 
                casted_arrays["wind_directions"], 
                casted_arrays["source_coordinates"], 
                casted_arrays["emission_rates"], 
                casted_arrays["grid_coordinates"], 
                casted_arrays["sensor_coordinates"])

    def _check_wind_data(self, ws, skip_low_wind):
        if skip_low_wind:
            return

        if np.any(ws <= 0):
            raise( ValueError("[FastGaussianPuff] wind speeds must be greater than 0"))
        if np.any(ws < 1e-2):
            print("[FastGaussianPuff] WARNING: There's a wind speed < 0.01 m/s. This is likely a mistake and will cause slow performance. The simulation will continue, but results will be poor as the puff model is degenerate in low wind speeds.")

    def _ensure_utc(self, dt):
        """
        Ensures the input datetime is timezone-aware and in UTC.
        Converts to UTC if it's in a different timezone.
        """
        ts = pd.Timestamp(dt)

        if ts.tz is None:
            raise ValueError(
                f"[FastGaussianPuff] Naive datetime detected: {dt}. Please provide a timezone-aware datetime."
            )

        if ts.tz != datetime.timezone.utc:
            ts = ts.tz_convert("UTC")

        return ts

    def _interpolate_wind_data(self, wind_speeds, wind_directions, puff_dt, sim_start, sim_end):
        '''
        Resample wind_speeds and wind_directions to the simulation resolution by interpolation.
        Inputs:
            sim_dt [s] (int): 
                the target time resolution to resample to
            sim_start, sim_end (pd.DateTime)
                DateTimes the simulation start and ends at
            n_obs (int)
                number of observation data points across the simulation time range
        '''

        # creates a timeseries at obs_dt resolution
        time_stamps = pd.date_range(sim_start, sim_end, self.n_obs)

        # technically need a final value to interpolate between, so just extend timeseries
        if len(wind_speeds) == self.n_obs - 1:
            wind_speeds = np.append(wind_speeds, wind_speeds[-1])

        if len(wind_directions) == self.n_obs - 1:
            wind_directions = np.append(wind_directions, wind_directions[-1])

        # interpolate for wind_speeds and wind_directions:
        ## 1. convert ws & wd to x and y component of wind (denoted by u, v, respectively)
        ## 2. interpolate u and v
        ## 3. bring resampled u and v back to resampled ws and wd
        wind_u, wind_v = self._wind_vector_convert(wind_speeds, 
                                                    wind_directions,
                                                    direction = 'ws_wd_2_u_v')

        # resamples wind data to sim_dt resolution
        wind_df = pd.DataFrame(data = {'wind_u' : wind_u,
                                    'wind_v' : wind_v}, 
                            index = time_stamps)

        wind_df = wind_df.resample(str(puff_dt)+'s').interpolate()
        wind_u = wind_df['wind_u'].to_list()
        wind_v = wind_df['wind_v'].to_list()

        self.wind_speeds_sim, self.wind_directions_sim = self._wind_vector_convert(wind_u, wind_v,direction= 'u_v_2_ws_wd') 

    def _wind_vector_convert(self, input_1, input_2, direction = 'ws_wd_2_u_v'):
        '''
        Convert between (ws, wd) and (u,v)
        Inputs:
            input_1: 
                - list of wind speed [m/s] if direction = 'ws_wd_2_u_v'
                - list of the x component (denoted by u) of a wind vector [m/s] if direction = 'u_v_2_ws_wd'
            input_2: 
                - list of wind direction [degree] if direction = 'ws_wd_2_u_v'
                - list of the y component (denoted by v) of a wind vector [m/s] if direction = 'u_v_2_ws_wd'
            direction: 
                - 'ws_wd_2_u_v': convert wind speed and wind direction to x,y components of a wind vector
                - 'u_v_2_ws_wd': convert x,y components of a wind vector to wind speed and wind directin
                       
        Outputs:
            quantities corresponding to the conversion direction
        '''
        if direction == 'ws_wd_2_u_v':
            ws, wd = input_1, input_2
            thetas = [270-x for x in wd] # convert wind direction to the angle between the wind vector and positive x-axis
            thetas = np.radians(thetas)
            u = [np.cos(theta)*l for l,theta in zip(ws, thetas)]
            v = [np.sin(theta)*l for l,theta in zip(ws, thetas)]
            output_1, output_2 = u, v
        elif direction == 'u_v_2_ws_wd':
            u, v = input_1, input_2
            ws = [(x**2 + y**2)**(1/2) for x,y in zip(u,v)]
            thetas = [np.arctan2(y, x) for x,y in zip(u,v)] # the angles between the wind vector and positive x-axis
            thetas = [x * 180 / np.pi for x in thetas]
            wd = [270 - x for x in thetas] # convert back to cardinal definition
            for i, x in enumerate(wd):
                if x < 0:
                    wd[i] = wd[i] + 360 
                elif x >= 360:
                    wd[i] = wd[i] - 360 
                else:
                    pass
            output_1, output_2 = ws, wd
        else:
            raise NotImplementedError(">>>>> wind vector conversion direction")

        return output_1, output_2

    def _model_info_print(self):
        '''
        Print the parameters used in this model
        '''

        print("\n************************************************************")
        print("****************     PUFF SIMULATION START     *************")
        print("************************************************************")
        print(">>>>> start time: {}".format(datetime.datetime.now()))
        print(">>>>> configuration;")
        print("         Observation time resolution: {}[s]".format(self.obs_dt))
        print("         Simulation time resolution: {}[s]".format(self.sim_dt))
        print("         Puff creation time resolution: {}[s]".format(self.puff_dt))
        if self.using_sensors:
            print("         Running in sensor mode")
        else:
            print(f"         Running in grid mode with grid dimensions {self.grid_dims}")

    def simulate(self):
        '''
        Main code for simulation
        Outputs:
            ch4_sim_res [ppm] (2-D np.array, shape = [N_t_obs, N_sensor]): 
                simulated concentrations resampled according to observation dt
        '''
        if self.quiet == False:
            self._model_info_print()

        self.GPC.simulate(self.ch4_sim)

        # resample results to the output_dt-resolution
        self.ch4_obs = self._resample_simulation(self.ch4_sim, self.out_dt)

        if self.quiet == False:
            print("\n************************************************************")
            print("*****************    PUFF SIMULATION END     ***************")
            print("************************************************************")

        return self.ch4_obs

    def _resample_simulation(self, c_matrix, resample_dt, mode = 'mean'):
        '''
        Resample the simulation results 
        Inputs:
            c_matrix [ppm] (2D np.ndarray, shape = [N_t_sim, self.N_points]): 
                the simulation results in sim_dt resolution across the whole grid
            dt [s] (scalar, float): 
                the target time resolution
            mode (str):
                - 'mean': resmple by taking average 
                - 'resample': resample by taking every dt sample
        Outputs:
            c_matrix_res [ppm] (4D np.array, shape = [N_t_new, self.grid_dims)]): 
                resampled simulation results 
        '''

        df = pd.DataFrame(c_matrix, index = self.time_stamps_sim)
        if mode == 'mean':
            df = df.resample(str(resample_dt)+'s').mean()
        elif mode == 'resample':
            df = df.resample(str(resample_dt)+'s').asfreq()
        else:
            raise NotImplementedError(">>>>> sim to obs resampling mode") 

        c_matrix_res = df.to_numpy()

        self.n_out = np.shape(c_matrix_res)[0]

        return c_matrix_res
