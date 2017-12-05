from nav.utility.classes import Location, Obstacle
from nav import solve

from benchmarks.point_to_point import problem
from benchmarks.point_to_point import environments

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

METHOD_ID = "graph_optimize_v1"
TEST_ID = 123
TEST_SIZE = 4

# Set problem variables
((x_min, y_min), (x_max, y_max)) = problem.area
boundary = [Location(x_min, y_min), Location(x_min, y_max),
            Location(x_max, y_min), Location(x_max, y_max)]

((wx_min, wy_min), (wx_max, wy_max)) = problem.waypoints
waypoints = [Location(wx_min, wy_min), Location(wx_max, wy_max)]

envs = environments.generate(TEST_SIZE, TEST_ID)
for i, env in enumerate(envs):
    stat_obstacles = []
    for obs in env:
        cx, cy, r = obs
        stat_obstacles.append(Obstacle(Location(cx, cy), r))

    try:
        flight_path = solve.point_to_point(boundary, waypoints, stat_obstacles)
    except Exception as e:
        print(e)
        flight_path = np.array([[wx_min, wx_max], [wy_min, wy_max]])

    np.save("{}_{}_{}.npy".format(METHOD_ID, TEST_ID, i), flight_path)

