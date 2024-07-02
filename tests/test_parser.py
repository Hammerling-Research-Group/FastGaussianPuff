import pytest
from FastGaussianPuff import PuffParser as parser
from FastGaussianPuff import GaussianPuff as GP
import pandas as pd
import numpy as np

def test_dlq_setup():
  p = parser('./parser_inputs/dlq.in')
  p.run_exp()

  # manually set up experiment and compare to parser output for 2 samples
  dat_dir = './parser_data/'
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
  start_time = pd.to_datetime("2022-02-01 09:59:11").floor('min')
  end_time = pd.to_datetime("2022-02-01 11:59:11").floor('min')

  wind_file = pd.read_csv(dat_dir + 'wind_median.csv')
  wind_file['timestamp'] = pd.to_datetime(wind_file['timestamp'])
  wind_dat = wind_file[(wind_file['timestamp'] >= start_time) & (wind_file['timestamp'] <= end_time)]
  ws = wind_dat['wind_speed'].values
  wd = wind_dat['wind_dir'].values
  
  gp = GP(60, 1.0, 1.0,
        start_time, end_time,
        source, [3.6],
        ws, wd,
        output_dt=60,
        using_sensors=True, sensor_coordinates=sensors)
  gp.simulate()

  parser_ch4 = pd.read_csv('./parser_out/01-02-22_09:59_exp_0.csv')

  for i in range(0, len(sensors)):
    diff = np.abs(parser_ch4[names[i]].values - gp.ch4_obs[:,i])
    assert np.linalg.norm(diff) < 1e-3

  source = [[488163.3384441765,4493892.532058168,5.447]]
  gp = GP(60, 1.0, 1.0,
        start_time, end_time,
        source, [3.6],
        ws, wd,
        output_dt=60,
        using_sensors=True, sensor_coordinates=sensors)
  gp.simulate()

  parser_ch4 = pd.read_csv('./parser_out/01-02-22_09:59_exp_3.csv')
  for i in range(0, len(sensors)):
    diff = np.abs(parser_ch4[names[i]].values - gp.ch4_obs[:,i])
    assert np.linalg.norm(diff) < 1e-3

if __name__ == '__main__':
  test_dlq_setup()