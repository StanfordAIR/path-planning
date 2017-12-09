from nav import solve

from benchmarks.point_to_point import problem
from benchmarks.point_to_point import environments
from benchmarks.point_to_point import draw
from benchmarks.point_to_point import benchmark

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle

######################################
# SPECIFY TEST
ENVIRONMENT_ID = 12345 # random seed for environments, defines the environment list
ENVIRONMENT_COUNT = 10 # number of environments in environment list envs
NUM_DISPLAY_ENVS = 9 # must be perfect square < ENVIRONMENT_COUNT
######################################

# Set problem variables
((x_min, y_min), (x_max, y_max)) = problem.area
((wx_min, wy_min), (wx_max, wy_max)) = problem.waypoints

# Solve environments
envs = environments.generate(ENVIRONMENT_COUNT, ENVIRONMENT_ID)
sols = []
for i, env in enumerate(envs):
    try:
        ######################################
        # Calculate path
        # The path should be a numpy array 
        #        [[x1, x2, x3, ..., xn],
        #         [y1, y2, y3, ..., yn]]
        # where (x1,y1) and (xn,yn) are the start and end points.
        flight_path = # your function to calculate the path, i.e func(env)
        #####################################
        
    except Exception as e:
        print(e)
        flight_path = np.array([[wx_min, wx_max], [wy_min, wy_max]])

    sols.append(flight_path)

# Display Solution Paths
draw.grid(NUM_DISPLAY_ENVS, envs, sols)

# Calculate benchmarks
benchmarks = benchmark.run(envs, sols)
print(benchmarks)
print("Score: {}".format(benchmarks["score"]))
