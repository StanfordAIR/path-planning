# -*- coding: utf-8 -*-
"""benchmark_environments.py
List of benchmark environments
"""

import random
from random import randint
from random import uniform
from random import gauss

from .problem import area
from .problem import waypoints
from .problem import obstacle_radius_mean
from .problem import obstacle_radius_variance
from .problem import max_obstacle_count

def inside(obs, waypoint):
    x = waypoint[0]
    y = waypoint[1]
    obs_x = obs[0]
    obs_y = obs[1]
    obs_r = obs[2]

    return (x - obs_x) ** 2 + (y - obs_y) ** 2 < obs_r ** 2

def generate(count, random_seed):
    random.seed(random_seed)
    envs = []
    while len(envs) < count:
        obs_count = randint(1, max_obstacle_count) # choose number of obstacles
        env = []
        while len(env) < obs_count:
            obs = (uniform(area[0][0], area[1][0]), # x from (x_min, x_max)
                   uniform(area[0][1], area[1][1]), # y from (y_min, y_max)
                   gauss(obstacle_radius_mean, obstacle_radius_variance))
            if (not inside(obs, waypoints[0]) and
                not inside(obs, waypoints[1]) and
                not obs[2] <=0):
                env.append(obs)
        envs.append(env)
    return envs

