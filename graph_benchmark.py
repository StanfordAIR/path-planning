from nav.utility.classes import Location, Obstacle
from nav import solve

from benchmarks.point_to_point import problem
from benchmarks.point_to_point import environments
from benchmarks.point_to_point import draw
from benchmarks.point_to_point import benchmark

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from pprint import pprint

######################################
# SPECIFY TEST
ENVIRONMENT_ID = 123456
ENVIRONMENT_COUNT = 16
NUM_DISPLAY_ENVS = 16 # must be perfect square < ENVIRONMENT_COUNT
######################################

params = pickle.load(open("graph_params.pkl", "rb"))
pprint(params)

# Set problem variables
((x_min, y_min), (x_max, y_max)) = problem.area
boundary = [Location(x_min, y_min), Location(x_min, y_max),
            Location(x_max, y_min), Location(x_max, y_max)]

((wx_min, wy_min), (wx_max, wy_max)) = problem.waypoints
waypoints = [Location(wx_min, wy_min), Location(wx_max, wy_max)]

# Solve environments
envs = environments.generate(ENVIRONMENT_COUNT, ENVIRONMENT_ID)
sols = []
for i, env in enumerate(envs):
    stat_obstacles = []
    for obs in env:
        cx, cy, r = obs
        stat_obstacles.append(Obstacle(Location(cx, cy), r))

    try:
        flight_path = solve.point_to_point(boundary, waypoints,
                                           stat_obstacles, params, verbose=False,
                                           display=False)
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
