import context

import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

import nav.env
import nav.plan
from nav.solve import point_to_point

RESULTS_DIR = 'tests/results/'
CONFIG_DIR = 'tests/config/'
BOUNDARY = np.genfromtxt(CONFIG_DIR + 'boundary.csv', delimiter=',').T
STATIC_OBS = np.genfromtxt(CONFIG_DIR + 'static_obs.csv', delimiter=',').T
PARAMS = {'granularity': 50, 'quantization_distance': 7}

WAYPOINTS = [(38.147,-76.434), (3.14345,-76.424)]

def test_environment_display():
    """ tests a simple environment display script for exceptions
    """
    result_dir = RESULTS_DIR + 'test_environment_display/'
    shutil.rmtree(result_dir, ignore_errors=True)
    os.makedirs(result_dir)

    env = nav.env.Environment(BOUNDARY, STATIC_OBS, PARAMS)
    #planner = nav.plan.Planner(env, PLANNER_PARAMS)
    
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    env.display(ax)
    fig.savefig(result_dir + 'env')

def test_planner_display():
    """ tests a simple planner display script for exceptions
    """
    result_dir = RESULTS_DIR + 'test_planner_display/'
    shutil.rmtree(result_dir, ignore_errors=True)
    os.makedirs(result_dir)

    # point_to_point(BOUNDARY, WAYPOINTS, STATIC_OBS, PARAMS, display=True)
