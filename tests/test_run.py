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

start = (7, 5)
end = (30, 14)

# start = (700, 500)
# end = (3000, 1400)

#I think granularity is how many ft per dot

def test_environment_display():
    """ tests a simple environment display script for exceptions
    """
    result_dir = RESULTS_DIR + 'test_environment_display/'
    shutil.rmtree(result_dir, ignore_errors=True)
    os.makedirs(result_dir)

    env = nav.env.Environment(BOUNDARY, STATIC_OBS, PARAMS)

    nodes = env.graph.base_nodes_ft 
    print(len(env.graph.base_nodes_ft[1]))
    print(len(env.graph.base_graph.nodes))
    print(nodes)
    max_x = np.max([i[0] for i in nodes])
    max_y = np.max([i[1] for i in nodes])
    print(max_x, max_y)

    # path = bidirectional_dijkstra(env.graph.base_graph, start, end, weight='weight')[1]
    path = shortest_path(env.graph.base_graph, start, end, weight='weight')
    # start = tuple(env.graph.base_nodes_ft[0])
    # end = tuple(env.graph.base_nodes_ft[-1])
    # path = bidirectional_dijkstra(env.graph.base_nodes_ft, start, end, weight='weight')[1]
    
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    env.display(ax)

    ax.plot([i[0] * env.graph.granularity for i in path], [i[1] * env.graph.granularity for i in path])
    print([i[0] * env.graph.granularity for i in path], [i[1] * env.graph.granularity for i in path])
    
    fig.savefig(result_dir + 'env')

test_environment_display()  