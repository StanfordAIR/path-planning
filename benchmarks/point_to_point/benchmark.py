# -*- coding: utf-8 -*-
"""benchmark.py
Measures performance of solutions to environments in environments
"""

import numpy as np

from .problem import area
from .problem import waypoints

def run(envs, sols):
    ((x_min, y_min), (x_max, y_max)) = area
    ((wx_min, wy_min), (wx_max, wy_max)) = waypoints
    # total length of paths
    total_length = 0
    total_obs_violations = 0
    total_bound_violations = 0
    total_good_solutions = 0
    for env, sol in zip(envs, sols):
        good_sol = True
        obs_violations = 0
        bound_violation = False

        # check start and endpoints
        epsilon = 0.5
        if (not close(sol[0,0], sol[1,0], wx_min, wy_min, epsilon)
            or not close(sol[0,-1], sol[1,-1], wx_max, wy_max, epsilon)):
            # if the start or endpoints do not match
            print("endpoints do not match")
            print("start: {},{}".format(sol[0,0], sol[0,1]))
            print("end  : {},{}".format(sol[-1,0], sol[-1,1]))
            good_sol = False

        # check for crashes
        for obs in env:
            c_x = obs[0]
            c_y = obs[1]
            r = obs[2]

            dist_x = sol[0] - c_x
            dist_y = sol[1] - c_y
            dist = np.sqrt(dist_x ** 2 + dist_y **2)
            if np.any(dist <= r):
                obs_violations += 1
                good_sol = False

        # check for boundary violation
        if (np.all(sol[0] < x_min) or np.all(sol[0] > x_max)
            or np.all(sol[1] < y_min) or np.all(sol[1] > y_max)):
	    # if a point is outside the flight area
            bound_violation = True
            good_sol = False

        if good_sol:
            total_length += np.sum(np.sqrt(np.sum(np.diff(sol) ** 2, axis=0)))
            print("benchmark, good sol")
        else: # if no solution was found add the length of the area perimiter
            print("benchmark, bad sol")
            total_length += 2 * (np.absolute(x_max - x_min)
                                 + np.absolute(y_max - y_min))
        total_obs_violations += obs_violations
        total_bound_violations += (1 if bound_violation else 0)
        total_good_solutions += (1 if good_sol else 0)


    avg_length = total_length / len(envs)
    avg_obs_violations = total_obs_violations / len(envs)
    avg_bound_violations = total_bound_violations / len(envs)
    success_rate = total_good_solutions / len(envs)
    benchmarks = {"score": avg_length,
                  "avg_length": avg_length,
                  "avg_obs_violations": avg_obs_violations,
                  "avg_bound_violations": avg_bound_violations,
                  "success_rate": success_rate}

    return benchmarks

def close(x1, y1, x2, y2, epsilon):
    dx = x2 - x1
    dy = y2 - y1
    dist = np.sqrt(dx ** 2 + dy ** 2)
    return dist < epsilon
