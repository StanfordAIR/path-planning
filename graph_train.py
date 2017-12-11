from itertools import product
import pickle

from nav.utility.classes import Location, Obstacle
from nav import solve

from benchmarks.point_to_point import problem
from benchmarks.point_to_point import environments
from benchmarks.point_to_point import benchmark

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

ENVIRONMENT_ID = 12345
ENVIRONMENT_COUNT = 12

params_values = {"granularity": [10.0],
                "quantization_distance": [1.0],
                "optim_max_iter": [800],
                "optim_init_step_size": [1e-3],
                "optim_min_step": [1e-2],
                "optim_reset_step_size": [1e-7, 1e-6],
                "optim_cooling_schedule": [1.0005, 1.0001],
                "optim_fast_cooling_schedule": [1.1, 1.3],
                "optim_init_constraint_hardness": [1, 2],
                "optim_init_spring_hardness": [.1, .2, .4],
                "optim_max_time_increase": [10], # what does this do?
                "optim_init_momentum":  [.99],
                "optim_momentum_change": [.1, .2],
                "optim_scale": [1.0]}

results_grid = []
score_grid = []
params_grid = [dict(zip(params_values, v)) for v in product(*params_values.values())]

# Set problem variables
((x_min, y_min), (x_max, y_max)) = problem.area
boundary = [Location(x_min, y_min), Location(x_min, y_max),
            Location(x_max, y_min), Location(x_max, y_max)]

((wx_min, wy_min), (wx_max, wy_max)) = problem.waypoints
waypoints = [Location(wx_min, wy_min), Location(wx_max, wy_max)]

for pn, params in enumerate(params_grid):
    print("trying params {} of {}".format(pn, len(params_grid)))
    envs = environments.generate(ENVIRONMENT_COUNT, ENVIRONMENT_ID)
    flight_paths = []
    for i, env in enumerate(envs):
        print("env {}".format(i))
        stat_obstacles = []
        for obs in env:
            cx, cy, r = obs
            stat_obstacles.append(Obstacle(Location(cx, cy), r))
    
        try:
            flight_path = solve.point_to_point(boundary, waypoints,
                                               stat_obstacles, params, display=False, verbose=False)
        except Exception as e:
            print(e)
            flight_path = np.array([[wx_min, wx_max], [wy_min, wy_max]])
    
        flight_paths.append(flight_path)

    result = benchmark.run(envs, flight_paths)
    results_grid.append(result)
    score_grid.append(result["score"])

best_params = params_grid[score_grid.index(min(score_grid))]
pickle.dump(best_params, open("graph_params.pkl", "wb"))
