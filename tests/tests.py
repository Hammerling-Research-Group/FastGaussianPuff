import time
import sys
import pandas as pd
import numpy as np

code_dir = '../'
sys.path.insert(0, code_dir)

from utilities import wind_synthesizer
from GaussianPuff import GaussianPuff as GP

# source: 4T-11
start_1 = '2022-02-22 01:33:22'
end_1 = '2022-02-22 03:33:23'

# source: 5S-27
start_2 = '2022-02-26 21:36:00'
end_2 = '2022-02-26 23:07:00'

# source: 4W-47
start_3 = '2022-04-27 03:49:09'
end_3 = '2022-04-27 08:04:09'

num_tests = 0
tests_passed = 0
tests_failed = 0
failed_tests = []


# Load in data
data_dir = './test_data/'

# 1-minute resolution wind data
df_ws_1min = pd.read_csv(data_dir + 'df_ws_1min_METEC_ADET.csv') 
df_wd_1min = pd.read_csv(data_dir + 'df_wd_1min_METEC_ADET.csv')
df_ws_1min['time_stamp.mountain'] = pd.to_datetime(df_ws_1min['time_stamp.mountain'])
df_wd_1min['time_stamp.mountain'] = pd.to_datetime(df_wd_1min['time_stamp.mountain'])


# Data processing
# column names used in the load in dfs
colnames = {'name' : 'name', 
            'x' : 'utm_easting.m',
            'y' : 'utm_northing.m',
            'z' : 'height.m',
            't' : 'time_stamp.mountain',
        'exp_id' : 'experiment_id', 
        'exp_t_0' : 'start_time.mountain', 
    'exp_t_end' : 'end_time.mountain', 
'emission_rate' : 'emission_rate.kg/hr'}

# synethize wind data- combines wind data from multiple sensors into one timeseries
if df_ws_1min.shape == df_wd_1min.shape:
    wind_syn_mode, wind_sensor = 'circular_mean', None
    ws_syn, wd_syn = wind_synthesizer(df_ws_1min, df_wd_1min, 
                                    wind_syn_mode, wind_sensor = wind_sensor,
                                    colname_t = colnames['t'])
    time_stamp_wind = df_ws_1min[colnames['t']].to_list()
else:
    raise ValueError(">>>>> df_ws and df_wd must have the same shape.") 

def runSensorTest(exp_start, t_0, t_end, 
            wind_speeds, wind_directions, 
            obs_dt, sim_dt, puff_dt,
            sensor_coordinates, source_coordinates, 
            emission_rate, puff_duration, 
            unsafe=False
            ):
    
    eps = 1e-7

    sensor_puff = GP(obs_dt, sim_dt, puff_dt,
                t_0, t_end,
                source_coordinates, emission_rate,
                wind_speeds, wind_directions, 
                using_sensors=True,
                sensor_coordinates=sensor_coordinates,
                quiet=True,
                puff_duration=puff_duration, unsafe=unsafe,
                exp_threshold_tolerance=eps
    )

    start = time.time()
    ch4 = sensor_puff.simulate()
    end = time.time()
    print("Runtime: ", end-start)

    # compare to ground truth, generated using original code
    test_data_dir = "./test_data/old/"
    start_time_str = exp_start.replace(" ", "-").replace(":", "-")
    filename = test_data_dir + "ch4-sensor-n-" + str(sensor_puff.N_points) + "-sim-" + str(sim_dt) + "-puff-" + str(puff_dt) + "-exp-" + start_time_str + ".csv"
    ch4_old = np.loadtxt(filename, delimiter=",")

    return check_test(ch4_old, ch4, unsafe)

def runTest(exp_start, t_0, t_end, 
            wind_speeds, wind_directions, 
            obs_dt, sim_dt, puff_dt,
            nx, ny, nz, 
            source_coordinates, emission_rate, grid_coords, puff_duration,
            unsafe = False
            ):
    
    if unsafe:
        eps = 1e-5
    else:
        eps = 1e-7

    grid_puff = GP(obs_dt, sim_dt, puff_dt,
                t_0, t_end,
                source_coordinates, emission_rate,
                wind_speeds, wind_directions, 
                grid_coordinates=grid_coords,
                using_sensors=False,
                nx=nx, ny=ny, nz=nz,
                quiet=True, unsafe=unsafe,
                puff_duration=puff_duration,
                exp_threshold_tolerance=eps
    )

    start = time.time()
    ch4 = grid_puff.simulate()
    end = time.time()
    print("Runtime: ", end-start)

    # compare to ground truth, generated using original code
    test_data_dir = "./test_data/old/"
    start_time_str = exp_start.replace(" ", "-").replace(":", "-")
    filename = test_data_dir + "ch4-n-" + str(grid_puff.N_points) + "-sim-" + str(sim_dt) + "-puff-" + str(puff_dt) + "-exp-" + start_time_str + ".csv"
    ch4_old = np.loadtxt(filename, delimiter=",")

    return check_test(ch4_old, ch4, unsafe=unsafe)

def check_test(ch4_old, ch4, unsafe=False):

    passed = True

    # unsafe tests have a more lenient error bound since the unsafe exponential approximation has a ~3% max error
    if unsafe:
        tol = 0.05
    else:
        tol = 0.001

    # stop one step short of end: original code doesn't actually produce results for final timestep, so skip it
    for t in range(0, len(ch4_old)-1):

        # if everything is close to or equal to zero, we don't care.
        if np.max(ch4_old[t]) < 1e-3:
            continue

        max_err = np.max(ch4_old[t].ravel() - ch4[t].ravel())/np.max(ch4_old[t])

        if np.isnan(max_err):
            print(f"ERROR: NAN present ch4 array at time {t}")
            if passed:
                passed = False
        if max_err > tol: # inequality doesn't work if there are NAN's
            # print("TIME: ", t)
            print("max err: ", max_err)

            if passed:
                passed = False # prevent from printing repeatedly
                print(f"ERROR: Max difference from expected value is greater than {tol*100}%")

    return passed

def general_tests():
    ################## GENERAL TESTS ###################

    x_num = 20
    y_num = 20
    z_num = 20

    puff_duration = 1080 # used in the original python code

    obs_dt, sim_dt, puff_dt = 60, 1, 1 # [seconds]
    grid_coords = [488098.55629668134, 4493841.098107514, 0, 488237.6735969247, 4493956.159806994, 24.0]

    ################ TEST 1 ######################
    global num_tests
    global tests_passed, tests_failed, failed_tests
    num_tests += 1

    source_coordinates = [[488163.3384441765, 4493892.532058168, 4.5]]
    emission_rate = [1.953021587640098]

    exp_start = pd.to_datetime(start_1)
    exp_end = pd.to_datetime(end_1)

    t_0 = exp_start.floor('min')
    t_end = exp_end.floor('min')

    idx_0 = pd.Index(time_stamp_wind).get_indexer([exp_start], method='nearest')[0]
    idx_end = pd.Index(time_stamp_wind).get_indexer([exp_end], method='nearest')[0]
    wind_speeds = ws_syn[idx_0 : idx_end+1]
    wind_directions = wd_syn[idx_0 : idx_end+1]

    print("-----------------------------------------")
    print("RUNNING TEST ", num_tests)
    passed = runTest(start_1, t_0, t_end, wind_speeds, wind_directions, 
                    obs_dt, sim_dt, puff_dt, x_num, y_num, z_num, 
                    source_coordinates, emission_rate, grid_coords, puff_duration)

    if not passed:
        print ("ERROR: TEST " + str(num_tests) + " FAILED")
        tests_failed += 1
    else:
        print ("Test " + str(num_tests) + " passed")
        tests_passed += 1

    ################ TEST 2 ######################
    num_tests += 1

    x_num = 20
    y_num = 20
    z_num = 20

    source_coordinates = [[488206.3525776105, 4493911.77819326, 2.0]]
    emission_rate = [0.8436203738042646]

    exp_start = pd.to_datetime(start_2)
    exp_end = pd.to_datetime(end_2)

    t_0 = exp_start.floor('min')
    t_end = exp_end.floor('min')

    idx_0 = pd.Index(time_stamp_wind).get_indexer([exp_start], method='nearest')[0]
    idx_end = pd.Index(time_stamp_wind).get_indexer([exp_end], method='nearest')[0]
    wind_speeds = ws_syn[idx_0 : idx_end+1]
    wind_directions = wd_syn[idx_0 : idx_end+1]

    print("-----------------------------------------")
    print("RUNNING TEST ", num_tests)
    passed = runTest(start_2, t_0, t_end, wind_speeds, wind_directions, 
                    obs_dt, sim_dt, puff_dt, x_num, y_num, z_num, 
                    source_coordinates, emission_rate, grid_coords, puff_duration)

    if not passed:
        print ("ERROR: TEST " + str(num_tests) + " FAILED")
        tests_failed += 1
    else:
        print ("Test " + str(num_tests) + " passed")
        tests_passed += 1

    ################ TEST 3 ######################
    num_tests += 1

    x_num = 20
    y_num = 20
    z_num = 20

    source_coordinates = [[488124.41821990383, 4493915.016403197, 2.0]]
    emission_rate = [0.5917636636467585]

    exp_start = pd.to_datetime(start_3)
    exp_end = pd.to_datetime(end_3)

    t_0 = exp_start.floor('min')
    t_end = exp_end.floor('min')

    idx_0 = pd.Index(time_stamp_wind).get_indexer([exp_start], method='nearest')[0]
    idx_end = pd.Index(time_stamp_wind).get_indexer([exp_end], method='nearest')[0]
    wind_speeds = ws_syn[idx_0 : idx_end+1]
    wind_directions = wd_syn[idx_0 : idx_end+1]

    print("-----------------------------------------")
    print("RUNNING TEST ", num_tests)
    passed = runTest(start_3, t_0, t_end, wind_speeds, wind_directions, 
                    obs_dt, sim_dt, puff_dt, x_num, y_num, z_num, 
                    source_coordinates, emission_rate, grid_coords, puff_duration)

    if not passed:
        print ("ERROR: TEST " + str(num_tests) + " FAILED")
        tests_failed += 1
    else:
        print ("Test " + str(num_tests) + " passed")
        tests_passed += 1

def non_square_tests():
    ################## NON_SQUARE GRID TESTS ###################
    global num_tests
    global tests_passed, tests_failed, failed_tests

    puff_duration = 1080 # used in the original python code

    obs_dt, sim_dt, puff_dt = 60, 1, 1 # [seconds]
    grid_coords = [488098.55629668134, 4493841.098107514, 0, 488237.6735969247, 4493956.159806994, 24.0]
    
    ################ TEST 1 ######################
    num_tests += 1

    # 400
    x_num = 10
    y_num = 5
    z_num = 8

    source_coordinates = [[488163.3384441765, 4493892.532058168, 4.5]]
    emission_rate = [1.953021587640098]

    exp_start = pd.to_datetime(start_1)
    exp_end = pd.to_datetime(end_1)

    t_0 = exp_start.floor('min')
    t_end = exp_end.floor('min')

    idx_0 = pd.Index(time_stamp_wind).get_indexer([exp_start], method='nearest')[0]
    idx_end = pd.Index(time_stamp_wind).get_indexer([exp_end], method='nearest')[0]
    wind_speeds = ws_syn[idx_0 : idx_end+1]
    wind_directions = wd_syn[idx_0 : idx_end+1]

    print("-----------------------------------------")
    print("RUNNING TEST ", num_tests)
    passed = runTest(start_1, t_0, t_end, wind_speeds, wind_directions, 
                    obs_dt, sim_dt, puff_dt, x_num, y_num, z_num, 
                    source_coordinates, emission_rate, grid_coords, puff_duration)
    
    if not passed:
        print ("ERROR: TEST " + str(num_tests) + " FAILED")
        tests_failed += 1
    else:
        print ("Test " + str(num_tests) + " passed")
        tests_passed += 1

    ################ TEST 2 ######################
    num_tests += 1

    # 816
    x_num = 12
    y_num = 17
    z_num = 4

    source_coordinates = [[488206.3525776105, 4493911.77819326, 2.0]]
    emission_rate = [0.8436203738042646]

    exp_start = pd.to_datetime(start_2)
    exp_end = pd.to_datetime(end_2)

    t_0 = exp_start.floor('min')
    t_end = exp_end.floor('min')

    idx_0 = pd.Index(time_stamp_wind).get_indexer([exp_start], method='nearest')[0]
    idx_end = pd.Index(time_stamp_wind).get_indexer([exp_end], method='nearest')[0]
    wind_speeds = ws_syn[idx_0 : idx_end+1]
    wind_directions = wd_syn[idx_0 : idx_end+1]

    print("-----------------------------------------")
    print("RUNNING TEST ", num_tests)
    passed = runTest(start_2, t_0, t_end, wind_speeds, wind_directions, 
                    obs_dt, sim_dt, puff_dt, x_num, y_num, z_num, 
                    source_coordinates, emission_rate, grid_coords, puff_duration)
    
    if not passed:
        print ("ERROR: TEST " + str(num_tests) + " FAILED")
        tests_failed += 1
    else:
        print ("Test " + str(num_tests) + " passed")
        tests_passed += 1

def varying_timestep_tests():

    global num_tests
    global tests_passed, tests_failed, failed_tests

    puff_duration = 1080 # used in the original python code

    grid_coords = [488098.55629668134, 4493841.098107514, 0, 488237.6735969247, 4493956.159806994, 24.0]

    ################ TEST 1 ######################
    num_tests += 1

    x_num = 20
    y_num = 20
    z_num = 20

    obs_dt, sim_dt, puff_dt = 60, 10, 60 # [seconds]

    source_coordinates = [[488163.3384441765, 4493892.532058168, 4.5]]
    emission_rate = [1.953021587640098]

    exp_start = pd.to_datetime(start_1)
    exp_end = pd.to_datetime(end_1)

    t_0 = exp_start.floor('min')
    t_end = exp_end.floor('min')

    idx_0 = pd.Index(time_stamp_wind).get_indexer([exp_start], method='nearest')[0]
    idx_end = pd.Index(time_stamp_wind).get_indexer([exp_end], method='nearest')[0]
    wind_speeds = ws_syn[idx_0 : idx_end+1]
    wind_directions = wd_syn[idx_0 : idx_end+1]

    print("-----------------------------------------")
    print("RUNNING TEST ", num_tests)
    passed = runTest(start_1, t_0, t_end, wind_speeds, wind_directions, 
                    obs_dt, sim_dt, puff_dt, x_num, y_num, z_num, 
                    source_coordinates, emission_rate, grid_coords, puff_duration)

    if not passed:
        print ("ERROR: TEST " + str(num_tests) + " FAILED")
        tests_failed += 1
    else:
        print ("Test " + str(num_tests) + " passed")
        tests_passed += 1

  ################ TEST 2 ######################
    num_tests += 1

    # 1620
    x_num = 15
    y_num = 12
    z_num = 9

    obs_dt, sim_dt, puff_dt = 60, 5, 30 # [seconds]

    source_coordinates = [[488206.3525776105, 4493911.77819326, 2.0]]
    emission_rate = [0.8436203738042646]

    exp_start = pd.to_datetime(start_2)
    exp_end = pd.to_datetime(end_2)

    t_0 = exp_start.floor('min')
    t_end = exp_end.floor('min')

    idx_0 = pd.Index(time_stamp_wind).get_indexer([exp_start], method='nearest')[0]
    idx_end = pd.Index(time_stamp_wind).get_indexer([exp_end], method='nearest')[0]
    wind_speeds = ws_syn[idx_0 : idx_end+1]
    wind_directions = wd_syn[idx_0 : idx_end+1]

    print("-----------------------------------------")
    print("RUNNING TEST ", num_tests)
    passed = runTest(start_2, t_0, t_end, wind_speeds, wind_directions, 
                    obs_dt, sim_dt, puff_dt, x_num, y_num, z_num, 
                    source_coordinates, emission_rate, grid_coords, puff_duration)

    if not passed:
        print ("ERROR: TEST " + str(num_tests) + " FAILED")
        tests_failed += 1
    else:
        print ("Test " + str(num_tests) + " passed")
        tests_passed += 1

    return

def sensor_tests():
 ################## GENERAL TESTS ###################
    global num_tests
    global tests_passed, tests_failed, failed_tests

    puff_duration = 1080 # used in the original python code

    obs_dt, sim_dt, puff_dt = 60, 1, 1 # [seconds]

    ################ TEST 1 ######################
    # uses real wind data with fabricated emission sources
    num_tests += 1

    source_coordinates = [[10.5, 23, 4.5]]
    emission_rate = [1.953021587640098]
    sensor_coordinates = [[0, 1, 3.5], 
                            [12.3, 4.9, 8],
                            [-11, -4, 2],
                            [-20, 5, 2],
                            [14, -7, 4]]
    

    exp_start = pd.to_datetime(start_1)
    exp_end = pd.to_datetime(end_1)

    t_0 = exp_start.floor('min')
    t_end = exp_end.floor('min')

    idx_0 = pd.Index(time_stamp_wind).get_indexer([exp_start], method='nearest')[0]
    idx_end = pd.Index(time_stamp_wind).get_indexer([exp_end], method='nearest')[0]
    wind_speeds = ws_syn[idx_0 : idx_end+1]
    wind_directions = wd_syn[idx_0 : idx_end+1]

    print("-----------------------------------------")
    print("RUNNING TEST ", num_tests)
    passed = runSensorTest(start_1, t_0, t_end,
                           wind_speeds, wind_directions, 
                           obs_dt, sim_dt, puff_dt,
                           sensor_coordinates, source_coordinates, 
                           emission_rate, puff_duration)

    if not passed:
        print ("ERROR: TEST " + str(num_tests) + " FAILED")
        tests_failed += 1
    else:
        print ("Test " + str(num_tests) + " passed")
        tests_passed += 1

    
    ################ TEST 2 ######################
    # uses sensor setup from METEC w real wind data
    num_tests += 1

    source_coordinates = [[488124.41821990383, 4493915.016403197, 2.0]]
    emission_rate = [0.5917636636467585]


    sensor_coordinates = [[488164.98285821447, 4493931.649887275, 2.4],
        [488198.08502694493, 4493932.618594243, 2.4],
        [488199.98542285635, 4493930.473245021, 2.4],
        [488226.9012860443, 4493887.916890612, 2.4],
        [488204.9825329503, 4493858.769131294, 2.4],
        [488172.4989330686, 4493858.565324413, 2.4],
        [488136.3904409793, 4493861.530987777, 2.4],
        [488106.145508258, 4493896.167438727, 2.4],
        [488133.15254321764, 4493932.355431944, 2.4],
        [488149.4733440923, 4493899.772560725, 2.4],
        [488156.6665182067, 4493895.099037761, 2.4]]
    

    exp_start = pd.to_datetime(start_3)
    exp_end = pd.to_datetime(end_3)

    t_0 = exp_start.floor('min')
    t_end = exp_end.floor('min')

    idx_0 = pd.Index(time_stamp_wind).get_indexer([exp_start], method='nearest')[0]
    idx_end = pd.Index(time_stamp_wind).get_indexer([exp_end], method='nearest')[0]
    wind_speeds = ws_syn[idx_0 : idx_end+1]
    wind_directions = wd_syn[idx_0 : idx_end+1]

    print("-----------------------------------------")
    print("RUNNING TEST ", num_tests)
    passed = runSensorTest(start_3, t_0, t_end,
                           wind_speeds, wind_directions, 
                           obs_dt, sim_dt, puff_dt,
                           sensor_coordinates, source_coordinates, 
                           emission_rate, puff_duration)

    if not passed:
        print ("ERROR: TEST " + str(num_tests) + " FAILED")
        tests_failed += 1
    else:
        print ("Test " + str(num_tests) + " passed")
        tests_passed += 1

    ################ TEST 3 ######################
    # same as test 3 but with different sim_dt and puff_dt
    num_tests += 1

    sim_dt = 5
    puff_dt = 30

    print("-----------------------------------------")
    print("RUNNING TEST ", num_tests)
    passed = runSensorTest(start_3, t_0, t_end,
                           wind_speeds, wind_directions, 
                           obs_dt, sim_dt, puff_dt,
                           sensor_coordinates, source_coordinates, 
                           emission_rate, puff_duration)

    if not passed:
        print ("ERROR: TEST " + str(num_tests) + " FAILED")
        tests_failed += 1
    else:
        print ("Test " + str(num_tests) + " passed")
        tests_passed += 1

def unsafe_tests():
    x_num = 20
    y_num = 20
    z_num = 20

    unsafe = True

    puff_duration = 1080 # used in the original python code

    obs_dt, sim_dt, puff_dt = 60, 1, 1 # [seconds]
    grid_coords = [488098.55629668134, 4493841.098107514, 0, 488237.6735969247, 4493956.159806994, 24.0]

    ################ TEST 1 ######################
    global num_tests
    global tests_passed, tests_failed, failed_tests
    num_tests += 1

    source_coordinates = [[488163.3384441765, 4493892.532058168, 4.5]]
    emission_rate = [1.953021587640098]

    exp_start = pd.to_datetime(start_1)
    exp_end = pd.to_datetime(end_1)

    t_0 = exp_start.floor('min')
    t_end = exp_end.floor('min')

    idx_0 = pd.Index(time_stamp_wind).get_indexer([exp_start], method='nearest')[0]
    idx_end = pd.Index(time_stamp_wind).get_indexer([exp_end], method='nearest')[0]
    wind_speeds = ws_syn[idx_0 : idx_end+1]
    wind_directions = wd_syn[idx_0 : idx_end+1]

    print("-----------------------------------------")
    print("RUNNING TEST ", num_tests)
    passed = runTest(start_1, t_0, t_end, wind_speeds, wind_directions, 
                    obs_dt, sim_dt, puff_dt, x_num, y_num, z_num, 
                    source_coordinates, emission_rate, grid_coords, puff_duration, unsafe)

    if not passed:
        print ("ERROR: TEST " + str(num_tests) + " FAILED")
        tests_failed += 1
    else:
        print ("Test " + str(num_tests) + " passed")
        tests_passed += 1

    ################ TEST 2 ######################
    # uses sensor setup from METEC w real wind data
    num_tests += 1

    source_coordinates = [[488124.41821990383, 4493915.016403197, 2.0]]
    emission_rate = [0.5917636636467585]


    sensor_coordinates = [[488164.98285821447, 4493931.649887275, 2.4],
        [488198.08502694493, 4493932.618594243, 2.4],
        [488199.98542285635, 4493930.473245021, 2.4],
        [488226.9012860443, 4493887.916890612, 2.4],
        [488204.9825329503, 4493858.769131294, 2.4],
        [488172.4989330686, 4493858.565324413, 2.4],
        [488136.3904409793, 4493861.530987777, 2.4],
        [488106.145508258, 4493896.167438727, 2.4],
        [488133.15254321764, 4493932.355431944, 2.4],
        [488149.4733440923, 4493899.772560725, 2.4],
        [488156.6665182067, 4493895.099037761, 2.4]]
    

    exp_start = pd.to_datetime(start_3)
    exp_end = pd.to_datetime(end_3)

    t_0 = exp_start.floor('min')
    t_end = exp_end.floor('min')

    idx_0 = pd.Index(time_stamp_wind).get_indexer([exp_start], method='nearest')[0]
    idx_end = pd.Index(time_stamp_wind).get_indexer([exp_end], method='nearest')[0]
    wind_speeds = ws_syn[idx_0 : idx_end+1]
    wind_directions = wd_syn[idx_0 : idx_end+1]

    print("-----------------------------------------")
    print("RUNNING TEST ", num_tests)
    passed = runSensorTest(start_3, t_0, t_end,
                           wind_speeds, wind_directions, 
                           obs_dt, sim_dt, puff_dt,
                           sensor_coordinates, source_coordinates, 
                           emission_rate, puff_duration, unsafe)

    if not passed:
        print ("ERROR: TEST " + str(num_tests) + " FAILED")
        tests_failed += 1
    else:
        print ("Test " + str(num_tests) + " passed")
        tests_passed += 1

    ################ TEST 3 ######################
    num_tests += 1

    x_num = 20
    y_num = 20
    z_num = 20

    source_coordinates = [[488124.41821990383, 4493915.016403197, 2.0]]
    emission_rate = [0.5917636636467585]

    exp_start = pd.to_datetime(start_3)
    exp_end = pd.to_datetime(end_3)

    t_0 = exp_start.floor('min')
    t_end = exp_end.floor('min')

    idx_0 = pd.Index(time_stamp_wind).get_indexer([exp_start], method='nearest')[0]
    idx_end = pd.Index(time_stamp_wind).get_indexer([exp_end], method='nearest')[0]
    wind_speeds = ws_syn[idx_0 : idx_end+1]
    wind_directions = wd_syn[idx_0 : idx_end+1]

    print("-----------------------------------------")
    print("RUNNING TEST ", num_tests)
    passed = runTest(start_3, t_0, t_end, wind_speeds, wind_directions, 
                    obs_dt, sim_dt, puff_dt, x_num, y_num, z_num, 
                    source_coordinates, emission_rate, grid_coords, puff_duration, unsafe=True)

    if not passed:
        print ("ERROR: TEST " + str(num_tests) + " FAILED")
        tests_failed += 1
    else:
        print ("Test " + str(num_tests) + " passed")
        tests_passed += 1


print("\nRUNNING GENERAL TESTS")
general_tests()
print("\nRUNNING UNSAFE MODE TESTS")
unsafe_tests()
print("\nRUNNING NON-SQUARE TESTS")
non_square_tests()
print("\nRUNNING TIMESTEP TESTS")
varying_timestep_tests()
print("\nRUNNING SENSOR TESTS")
sensor_tests()


print("----------------------------------------------------------------")
print("RESULTS")
print("Total number of tests:", num_tests)
print("Tests passed:", tests_passed)
print("Tests failed:", tests_failed)
if tests_failed > 0:
    print("Test failed on test numbers:", failed_tests)
