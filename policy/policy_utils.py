# -*- coding: utf-8 -*-
"""
policy_utils.py
A shared library for all the tools for policy iteration.
"""

import numpy as np
import numpy.linalg as lin
import numpy.random as nprand

import matplotlib.pyplot as plt


MAP_TOP = 150
MAP_BOT = -50
PENALTY = -10**(-8)

START = np.zeros(2)
FINISH = 100 * np.ones(2)

NUM_OBS = 5
RADM = 15
RADSTD = 5

EPSILON = 10
TIME_CAP = 500

NUM_ACTIONS = 8
ACTIONS = [np.array((np.cos(theta), np.sin(theta))) for theta 
           in np.linspace(0, 2*np.pi, num=NUM_ACTIONS, endpoint=False)]


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


def gen_obstacles():
    """
    Generate obstacles across the map.
    """
    locs = nprand.randint(MAP_BOT, MAP_TOP, (NUM_OBS,2))
    rads = nprand.normal(RADM, RADSTD, NUM_OBS)
    
    for i in range(NUM_OBS):  # Remove invalid obstacles.
        if (lin.norm(locs[i]-START) <= rads[i] 
        or lin.norm(locs[i]-FINISH) <= rads[i]):
            locs[i] = MAP_BOT * np.ones(2)
            rads[i] = 1
    return locs, rads


def _validate_pos(pos, locs, rads):
    """
    Check if the position is not inside any obstacles. If so, return a new
    position.
    """
    while (_obs_reward(pos, locs, rads) < 0):
        pos = nprand.uniform(MAP_BOT, MAP_TOP, 2)
    return pos


def gen_training_states(samples):
    """
    Generate a random sample of states across the map.
    """
    states = []
    for sample in range(samples):
        locs, rads = gen_obstacles()
        pos = _validate_pos(nprand.uniform(MAP_BOT, MAP_TOP, 2), locs, rads)
        states += [(pos,locs,rads)]
    return states


"""
Testing and Visualization
    straight_policy
    test
    batch_test
    plot_performance
"""

def _circle(loc, rad, resolution=500):
    """ 
    Generate the x,y coordinates that define a circle.
    """
    t = np.linspace(0, 2*np.pi, resolution)
    return rad * np.cos(t) + loc[0] , rad * np.sin(t) + loc[1]


def _collision(pos, locs, rads):
    for i in range(len(locs)):
        if (lin.norm(pos - locs[i]) < rads[i]-0.5):
            return True
    return False


def straight_policy(pos, locs, rads, params):
    return (np.sqrt(2)/2) * np.ones(2)


def test(locs, rads, policy, params):
    """
    Test out a policy's performance on some map.
    """
    pos = START
    t = 0
    path = [pos]
    while(lin.norm(pos - FINISH) > EPSILON and t < TIME_CAP):
        pos = sim(pos, policy(pos, locs, rads, params))
        if(_collision(pos, locs, rads)):
            break
        path += [pos]
        t += 1
    
    success = 1 if lin.norm(pos - FINISH) < EPSILON else 0
    return success, path


def batch_test(samples, policy, params):
    """
    Test the policy on multiple examples and compare to a baseline.
    """
    model_count = 0
    baseline_count = 0
    test_data = gen_training_states(samples)
    for pos, locs, rads in test_data:
        model_count += test(locs, rads, policy, params)[0]
        baseline_count += test(locs, rads, straight_policy, None)[0]
    return round(model_count/samples, 2), round(baseline_count/samples, 2)
        

def plot_performance(path, locs, rads, figsize=8):
    """
    Plot a path and set of obstacles.
    """
    # Set up figure.
    plt.figure(figsize=(figsize,figsize))
    plt.ylim(MAP_BOT, MAP_TOP)
    plt.xlim(MAP_BOT, MAP_TOP)
    plt.grid()

    # Plot path.
    x = [pos[0] for pos in path]
    y = [pos[1] for pos in path]
    plt.scatter(x, y, marker='.', s=50)
    
    # Plot obstacles.
    for i in range(len(locs)):
        x, y = _circle(locs[i], rads[i])
        plt.scatter(x, y, marker='.', color='r', s=1)
        
    plt.show()
    
    
def sample_run(policy, params):
    """
    Generate a sample run and plot it for the user.
    """
    locs, rads = gen_obstacles()
    success, path = test(locs, rads, policy, params)
    plot_performance(path, locs, rads)
    
