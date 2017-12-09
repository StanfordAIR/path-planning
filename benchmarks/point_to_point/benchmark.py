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
    length = 0
    for env, sol in zip(envs, sols):
        length += np.sum(np.sqrt(np.sum(np.diff(sol) ** 2, axis=0)))

    # number of obstacle violations TODO add line segment check
    obs_violations = 0
    for env, sol in zip(envs, sols):
        for obs in env:
            c_x = obs[0]
            c_y = obs[1]
            r = obs[2]

            dist_x = sol[0] - c_x
            dist_y = sol[1] - c_y
            dist = np.sqrt(dist_x ** 2 + dist_y **2)
            if np.any(dist <= r):
                obs_violations += 1

    # number of boundary violations
    bound_violations = 0
    for env, sol in zip(envs, sols):
        if (np.all(sol[0] < x_min) or np.all(sol[0] > x_max)
            or np.all(sol[1] < y_min) or np.all(sol[1] > y_max)):
	    # if a point is outside the flight area
            bound_violations += 1


    score = length if obs_violations + bound_violations == 0 else 0
    benchmarks = {"score": score,
                  "length": length,
                  "obs_violations": obs_violations,
                  "bound_violations": bound_violations}

    return benchmarks
