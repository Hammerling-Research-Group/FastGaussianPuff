from FastGaussianPuff import PuffParser
import os
import pandas as pd
import numpy as np

def test_regression():
  dir = './regression/'
  p = PuffParser(dir + 'regression_test.in')
  p.run_exp()

  num_refs = 4
  num_comp = 0
  tol = 1e-6

  out_dir = dir + 'out/'
  ref_dir = dir + 'ref/'
  cols = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
  for filename in os.listdir(ref_dir):
    if not filename.endswith('.csv'):
      continue

    num_comp += 1
      
    ref_name = os.path.join(ref_dir, filename)
    test_name = os.path.join(out_dir, filename)

    ref = pd.read_csv(ref_name)
    test = pd.read_csv(test_name)

    for name in cols:
      assert np.allclose(ref[name].values, test[name].values, atol=tol)

  assert num_comp == num_refs


