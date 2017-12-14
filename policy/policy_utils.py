# -*- coding: utf-8 -*-
"""
policy_utils.py
A shared library for all the tools for policy iteration.
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lin
import pickle

MAP_TOP = 150
MAP_BOT = -50
PENALTY = -10**(-8)

START = np.zeros(2)
FINISH = 100 * np.ones(2)

NUM_ACTIONS = 9
ACTIONS = [np.array((np.cos(theta), np.sin(theta))) for theta 
           in np.linspace(0, 2*np.pi, num=NUM_ACTIONS-1, endpoint=False)]
ACTIONS += np.zeros(2)


"""
Utilities and Dynamics
Includes:
    closest
    sim
"""

def closest(approx):
    """
    Find the closest allowable action to an approximation of the action.
    """
    dists = [lin.norm(approx-act) for act in ACTIONS]
    return ACTIONS[np.argmin(dists)]


def sim(pos, action):
    """
    Generate the next state from a state, action pair.
    """
    return pos + action


"""
Training Tools
Includes:
    reward
    value
    gen_obstacles
    gen_training_states
"""

def _pos_reward(pos, pwr=2):
    """
    Reward for the distance from the objective.
    """
    return -lin.norm(pos-FINISH)**pwr


def _bnd_reward(pos):
    """
    Reward for staying in bounds.
    """
    out = (pos > MAP_TOP) + (pos < MAP_BOT)
    if (out.dot(out) > 0):
        return PENALTY
    return 0


def _obs_reward(pos, locs, rads):
    """
    Reward for not colliding with obstacles.
    """
    for i in range(len(locs)):
        if(lin.norm(pos-locs[i]) < rads[i]):
            return PENALTY
    return 0


def reward(pos, locs, rads):
    """
    Reward function.
    """
    return _pos_reward(pos) + _bnd_reward(pos) + _obs_reward(pos, locs, rads)


def value(pos, locs, rads, policy, params, discount=0.9, duration=15):
    """
    Generate the value estimate of the state.
    """
    cur = pos
    val = 0
    for i in range(duration):
        val += (discount**i) * reward(cur, locs, rads)
        cur = sim(cur, policy(cur, locs, rads, params))
    return val


def gen_obstacles(num=8, rad_mean=15, rad_std=5):
    """
    Generate obstacles over the map.
    """
    locs = np.random.randint(MAP_BOT, MAP_TOP, size=(num,2))
    rads = np.random.normal(rad_mean, rad_std, size=num)
    
    for i in range(num):  # Remove invalid obstacles.
        if (lin.norm(locs[i]-START) <= rads[i] 
        or lin.norm(locs[i]-FINISH) <= rads[i]):
            locs[i,:2] = -50 * np.ones(2)
            rads[i] = 1
    return locs, rads


def gen_training_states(samples):
    states = []
    # TODO

"""

"""