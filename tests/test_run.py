import context

import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

import nav.env
import nav.plan

from networkx.algorithms import shortest_path, bidirectional_dijkstra

RESULTS_DIR = 'tests/results/'
CONFIG_DIR = 'tests/config/'
BOUNDARY = np.genfromtxt(CONFIG_DIR + 'boundary.csv', delimiter=',').T
STATIC_OBS = np.genfromtxt(CONFIG_DIR + 'static_obs.csv', delimiter=',').T
PARAMS = {'granularity': 100, 'quantization_distance': 7}

# I think granularity is how many ft per dot
# I think it's (y, x)
# Maybe give more space around the obstacles when trimming just for safety

# workflow - given boundaries, obstacles, granularity
# build environment
# build path
# need to trim "in between" points
# and then convert to lat/long

# lat lng
current_position = (38.1425, -76.426)
target_position = (38.150, -76.4335)

def round_to_granularity(env, val):
    granularity = env.graph.granularity
    return int(val / granularity)

def build_path(env, cur, target):
    start = env.point_ll_to_ft(cur)
    end = env.point_ll_to_ft(target)
    start = (round_to_granularity(env, start[0]), round_to_granularity(env, start[1]))
    end = (round_to_granularity(env, end[0]), round_to_granularity(env, end[1]))
    print('s, e', start, end)
    path = shortest_path(env.graph.base_graph, start, end, weight='weight')
    # trim here if necessary

    #
    path = np.array([[p[1] for p in path], [p[0] for p in path]]) * 100.0
    path = env.ft_to_ll(path)
    return path

def test_environment_display():
    """ tests a simple environment display script for exceptions
    """
    result_dir = RESULTS_DIR + 'test_environment_display/'
    shutil.rmtree(result_dir, ignore_errors=True)
    os.makedirs(result_dir)

    env = nav.env.Environment(BOUNDARY, STATIC_OBS, PARAMS)

    start = (8, 27)
    end = (35, 5)
    path = build_path(env, current_position, target_position)

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    env.display(ax)
    # ax.plot([i[1] * env.graph.granularity for i in path], [i[0] * env.graph.granularity for i in path])
    path_ft = env.ll_to_ft(path)
    print(path_ft)
    ax.plot(path_ft[0], path_ft[1])
    fig.savefig(result_dir + 'env')

test_environment_display()  