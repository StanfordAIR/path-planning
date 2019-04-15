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

#I think granularity is how many ft per dot

# i think it's (y, x)

def build_path(env, start, end):
    return shortest_path(env.graph.base_graph, start, end, weight='weight')

def test_environment_display():
    """ tests a simple environment display script for exceptions
    """
    result_dir = RESULTS_DIR + 'test_environment_display/'
    shutil.rmtree(result_dir, ignore_errors=True)
    os.makedirs(result_dir)

    env = nav.env.Environment(BOUNDARY, STATIC_OBS, PARAMS)

    start = (8, 27)
    end = (35, 5)
    path = build_path(env, start, end)

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    env.display(ax)
    ax.plot([i[1] * env.graph.granularity for i in path], [i[0] * env.graph.granularity for i in path])
    fig.savefig(result_dir + 'env')

test_environment_display()  