# -*- coding: utf-8 -*-
"""benchmark_environments.py
List of benchmark environments
"""

import math
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

def on_beeline(env):
    ((wx_min, wy_min), (wx_max, wy_max)) = waypoints
    for obs in env:
    	if intersects_obs(wx_min, wy_min, wx_max, wy_max,
                            obs[0], obs[1], obs[2]):
            return True
    return False 
    
# based on a an answer by ryu jin at https://math.stackexchange.com/questions/275529/check-if-line-intersects-with-circles-perimeter
def intersects_obs(ax, ay, bx, by, cx, cy, r):
    distance = abs(((by - ay) * cx - (bx - ax) * cy + bx * ay - ax * by)
                   / math.sqrt((by - ay) ** 2 + (bx - ax) ** 2))
    return distance < r


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
            if (not inside(obs, waypoints[0])
                and not inside(obs, waypoints[1])
                and obs[2] > 0): # radius must be > 0
                env.append(obs)
        if on_beeline(env):
            envs.append(env)
    return envs

