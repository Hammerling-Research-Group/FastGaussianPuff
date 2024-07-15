import pandas as pd
import numpy as np
import ast
import os

from FastGaussianPuff import GaussianPuff as GP

class PuffParser:

  def __init__(self, input_file):
    self.input_file = input_file
    self.options = {}
    self.parse_options()

    parent_dir = os.path.dirname(self.input_file) + "/"

    self.source_file = pd.read_csv(parent_dir + self.options['source_file'])
    self.sensor_file = pd.read_csv(parent_dir + self.options['sensor_file'])

    self.wind_file = pd.read_csv(parent_dir + self.options['wind_file'])
    self.wind_file['timestamp'] = pd.to_datetime(self.wind_file['timestamp'], utc=True)
    
    self.exp_file = pd.read_csv(parent_dir + self.options['experiments_file'])
    self.output_dir = parent_dir + self.options['output_dir']

    self.colnames = ast.literal_eval(self.options['coordinate_columns'])

    self.parse_coordinates()

    self.wind_dt = float(self.options['wind_dt'])
    self.output_dt = float(self.options['output_dt'])
    self.sim_dt = float(self.options['sim_dt'])
    self.puff_dt = float(self.options['puff_dt'])

  def run_exp(self):
    start_times = self.exp_file['start_time'].values
    end_times = self.exp_file['end_time'].values
    n_exp = len(start_times)

    source_names = self.exp_file['sources'].values
    rates = self.exp_file['emission_rates'].values

    for i in range(n_exp):
      start_time = pd.to_datetime(start_times[i]).floor('min')
      end_time = pd.to_datetime(end_times[i]).floor('min')

      
      out_time_str = start_time.tz_localize(None).strftime('%m-%d-%y_%H:%M')
      out_fname = self.output_dir + out_time_str + '_exp_' + str(i) + '.csv'
      # out_name = self.output_dir + 'exp' + str(i) + '.csv'

      ws = self.wind_file[(self.wind_file['timestamp'] >= start_time) & (self.wind_file['timestamp'] <= end_time)]
      ws = ws['wind_speed'].values
      wd = self.wind_file[(self.wind_file['timestamp'] >= start_time) & (self.wind_file['timestamp'] <= end_time)]
      wd = wd['wind_dir'].values

      # C code expects no timezone
      start_time = start_time.tz_localize(None)
      end_time = end_time.tz_localize(None)

      exp_sources = ast.literal_eval(source_names[i])
      rate = ast.literal_eval(rates[i])

      n_sources = len(exp_sources)
      total_ch4 = 0

      sensors = list(self.sensor_coords.values())
      for j in range(n_sources):
        s_ind = exp_sources[j]
        coords = [self.source_coords[s_ind]]

        gp = GP(self.wind_dt, self.sim_dt, self.puff_dt,
                start_time, end_time,
                coords, [rate[j]],
                ws, wd,
                output_dt=self.output_dt,
                using_sensors=True, sensor_coordinates=sensors)
        gp.simulate()

        total_ch4 = gp.ch4_obs + total_ch4

      df_ch4 = pd.DataFrame(total_ch4, columns=list(self.sensor_coords.keys()))

      z = np.zeros((1, len(list(self.sensor_coords.keys()))))
      df_ch4 = pd.concat([pd.DataFrame(z, columns=list(self.sensor_coords.keys())), df_ch4], ignore_index=True)

      gp_out_res = gp.output_dt # seconds
      td = pd.Timedelta(gp_out_res, unit='seconds')
      
      time_series = pd.date_range(start_time, end_time + td, periods=gp.n_out + 1)
      df_ch4.insert(0, 'timestamp', time_series)
      pd.DataFrame(df_ch4).to_csv(out_fname, index=False)

  def parse_coordinates(self):
    self.coordinate_system = self.options['coordinate_system']
    if self.coordinate_system == 'geodetic':
      
      lat = self.source_file[self.colnames[1]].values
      lon = self.source_file[self.colnames[2]].values

      source_x, source_y = self.latlon_to_utm_array(lat, lon)

      lat = self.sensor_file[self.colnames[1]].values
      lon = self.sensor_file[self.colnames[2]].values

      sensor_x, sensor_y = self.latlon_to_utm_array(lat, lon)
    else:
      source_x = self.source_file[self.colnames[1]].values
      source_y = self.source_file[self.colnames[2]].values

      sensor_x = self.sensor_file[self.colnames[1]].values
      sensor_y = self.sensor_file[self.colnames[2]].values

    source_z = self.source_file[self.colnames[3]].values
    sensor_z = self.sensor_file[self.colnames[3]].values

    source_names = self.source_file[self.colnames[0]].values
    sensor_names = self.sensor_file[self.colnames[0]].values

    sources = {}
    sensors = {}
    for i,name in enumerate(source_names):
      sources[name] = [source_x[i],source_y[i], source_z[i]]

    for i,name in enumerate(sensor_names):
      sensors[name] = [sensor_x[i],sensor_y[i], sensor_z[i]]

    self.source_coords = sources
    self.sensor_coords = sensors

  def parse_options(self):
    comment_symbol = '#'
    with open(self.input_file, 'r') as f:
      for line in f:
        if comment_symbol in line: # Strip comments
          line, comment = line.split(comment_symbol, 1)

        line = line.strip()
        if line != '': # Save options to dictionary, Value may be more than one word
          option, value = line.split(maxsplit=1)

          self.options[option.lower()] = value

  def latlon_to_zone_number(self,latitude, longitude):
      if 56 <= latitude < 64 and 3 <= longitude < 12:
          return 32
      if 72 <= latitude <= 84 and longitude >= 0:
          if longitude <= 9:
              return 31
          elif longitude <= 21:
              return 33
          elif longitude <= 33:
              return 35
          elif longitude <= 42:
              return 37
      return int((longitude + 180) / 6) + 1

  ZONE_LETTERS = ["C", "D", "E", "F", "G", "H", "J",
          "K", "L", "M", "N", "P", "Q", "R",
          "S", "T", "U", "V", "W", "X", "X"]

  def latitude_to_zone_letter(self,latitude):
      latitude = int(latitude)
      if -80 <= latitude <= 84:
          return self.ZONE_LETTERS[(latitude + 80) >> 3]
      else:
          return None

  def zone_number_to_central_longitude(self,zone_number):
      return (zone_number - 1) * 6 - 180 + 3
  from math import pi, sin, cos, sqrt

  def latlon_to_utm_array(self,latitude, longitude):
    easting = np.zeros(latitude.shape)
    northing = np.zeros(latitude.shape)
    for i in range(len(latitude)):
      if np.isnan(latitude[i]) or np.isnan(longitude[i]):
        continue
      easting[i], northing[i], temp1, temp2 = self.latlon_to_utm(latitude[i], longitude[i])
    return easting, northing
     
  def latlon_to_utm(self,latitude, longitude, force_zone_number=None, R=6378137, E=0.00669438):
      # UTM scale on the central meridian
      K0 = 0.9996
      E2 = E * E
      E3 = E2 * E
      E_P2 = E / (1.0 - E)
      M1 = (1 - E / 4 - 3 * E2 / 64 - 5 * E3 / 256)
      M2 = (3 * E / 8 + 3 * E2 / 32 + 45 * E3 / 1024)
      M3 = (15 * E2 / 256 + 45 * E3 / 1024)
      M4 = (35 * E3 / 3072)
      lat_rad = np.pi * (latitude) / 180
      lat_sin = np.sin(lat_rad)
      lat_cos = np.cos(lat_rad)
      lat_tan = lat_sin / lat_cos
      lat_tan2 = lat_tan * lat_tan
      lat_tan4 = lat_tan2 * lat_tan2

      if force_zone_number is None:
          zone_number = self.latlon_to_zone_number(latitude, longitude)
      else:
          zone_number = force_zone_number

      zone_letter = self.latitude_to_zone_letter(latitude)
      lon_rad = np.pi * (longitude) / 180
      central_lon = self.zone_number_to_central_longitude(zone_number)
      central_lon_rad = np.pi * (central_lon) / 180

      n = R / np.sqrt(1 - E * lat_sin**2)
      c = E_P2 * lat_cos**2
      a = lat_cos * (lon_rad - central_lon_rad)
      a2 = a * a
      a3 = a2 * a
      a4 = a3 * a
      a5 = a4 * a
      a6 = a5 * a
      m = R * (M1 * lat_rad -
              M2 * np.sin(2 * lat_rad) +
              M3 * np.sin(4 * lat_rad) -
              M4 * np.sin(6 * lat_rad))
      easting = K0 * n * (a +
                          a3 / 6 * (1 - lat_tan2 + c) +
                          a5 / 120 * (5 - 18 * lat_tan2 + lat_tan4 + 72 * c - 58 * E_P2)) + 500000
      northing = K0 * (m + n * lat_tan * (a2 / 2 +
                                          a4 / 24 * (5 - lat_tan2 + 9 * c + 4 * c**2) +
                                          a6 / 720 * (61 - 58 * lat_tan2 + lat_tan4 + 600 * c - 330 * E_P2)))
      return easting, northing, zone_number, zone_letter
