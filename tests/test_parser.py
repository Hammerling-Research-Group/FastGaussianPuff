import pytest
from FastGaussianPuff import PuffParser as parser
from FastGaussianPuff import GaussianPuff as GP
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def test_dlq_setup():
  p = parser('./parser/in/dlq.in')
  p.run_exp()

  # manually set up experiment and compare to parser output for 2 samples
  dat_dir = './input_data/'
  sensors = [[488164.98285821447,4493931.649887275,2.4],
            [488198.08502694493,4493932.618594243,2.4],
              [488226.9012860443,4493887.916890612,2.4],
              [488204.9825329503,4493858.769131294,2.4],
              [488172.4989330686,4493858.565324413,2.4],
              [488136.3904409793,4493861.530987777,2.4],
              [488106.145508258,4493896.167438727,2.4],
              [488133.15254321764,4493932.355431944,2.4]]
  names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
  
  source = [[488205.23764607066,4493913.018740796,0.326]]
  start_time = pd.to_datetime("2022-02-01 09:59:11-07:00").floor('min')
  end_time = pd.to_datetime("2022-02-01 11:59:11-07:00").floor('min')

  wind_file = pd.read_csv(dat_dir + 'wind_median.csv')
  wind_file['timestamp'] = pd.to_datetime(wind_file['timestamp'], utc=True)
  wind_dat = wind_file[(wind_file['timestamp'] >= start_time) & (wind_file['timestamp'] <= end_time)]
  ws = wind_dat['wind_speed'].values
  wd = wind_dat['wind_dir'].values

  start_time = start_time.tz_localize(None)
  end_time = end_time.tz_localize(None)
  
  gp = GP(60, 1.0, 1.0,
        start_time, end_time,
        source, [3.6],
        ws, wd,
        output_dt=60,
        using_sensors=True, sensor_coordinates=sensors)
  gp.simulate()

  parser_ch4 = pd.read_csv('./parser/out/02-01-22_09:59_exp_0.csv')

  for i in range(0, len(sensors)):
    diff = np.abs(parser_ch4[names[i]].values[1:] - gp.ch4_obs[:,i])
    assert np.linalg.norm(diff) < 1e-3

  source = [[488163.3384441765,4493892.532058168,5.447]]
  gp = GP(60, 1.0, 1.0,
        start_time, end_time,
        source, [3.6],
        ws, wd,
        output_dt=60,
        using_sensors=True, sensor_coordinates=sensors)
  gp.simulate()

  parser_ch4 = pd.read_csv('./parser/out/02-01-22_09:59_exp_3.csv')
  for i in range(0, len(sensors)):
    diff = np.abs(parser_ch4[names[i]].values[1:] - gp.ch4_obs[:,i])
    assert np.linalg.norm(diff) < 1e-3

def test_multisource_setup():
  p = parser('./parser/in/multi.in')
  p.run_exp()


  # manually set up experiment and compare to parser output for 2 samples
  dat_dir = './input_data/'
  sensors = [[488164.98285821447,4493931.649887275,2.4],
            [488198.08502694493,4493932.618594243,2.4],
              [488226.9012860443,4493887.916890612,2.4],
              [488204.9825329503,4493858.769131294,2.4],
              [488172.4989330686,4493858.565324413,2.4],
              [488136.3904409793,4493861.530987777,2.4],
              [488106.145508258,4493896.167438727,2.4],
              [488133.15254321764,4493932.355431944,2.4]]
  names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

  wind_file = pd.read_csv(dat_dir + 'wind_median.csv')
  wind_file['timestamp'] = pd.to_datetime(wind_file['timestamp'], utc=True)
  
  start_time = pd.to_datetime("2022-04-08 10:22:21-06:00").floor('min')
  end_time = pd.to_datetime("2022-04-08 13:22:22-06:00").floor('min')

  wind_dat = wind_file[(wind_file['timestamp'] >= start_time) & (wind_file['timestamp'] <= end_time)]
  ws = wind_dat['wind_speed'].values
  wd = wind_dat['wind_dir'].values

  start_time = start_time.tz_localize(None)
  end_time = end_time.tz_localize(None)

  sources = [[488135.60300454,4493914.332570829,0.536], 
            [488205.72955104336,4493885.416201145,1.527]]
  rates = [[1.937954395777344],[2.049521188351958]]
  ch4 = 0
  
  for i, source in enumerate(sources):
    print(source, rates[i])
    gp = GP(60, 1.0, 1.0,
          start_time, end_time,
          [source], rates[i],
          ws, wd,
          output_dt=60,
          using_sensors=True, sensor_coordinates=sensors)
    gp.simulate()

    ch4 = gp.ch4_obs + ch4

  parser_ch4 = pd.read_csv('./parser/out/04-08-22_10:22_exp_0.csv')
  for i in range(0, len(sensors)):
    diff = np.abs(parser_ch4[names[i]].values[1:] - ch4[:,i])
    assert np.linalg.norm(diff) < 1e-3

if __name__ == '__main__':
  test_dlq_setup()