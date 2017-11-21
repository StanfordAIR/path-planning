# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 22:00:12 2017
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
S = np.array([[1,0,0],
             [0,1,0],
             [0,0,1]])

A = np.array([[1,0,0],
             [0,1,0],
             [0,0,1]])

TOP = 150
BOT = -50
PENALTY = 10**6

START = np.zeros(3)
OBJECTIVE = np.array([100,100,0])

R2 = np.sqrt(2)
ACTIONS = [[1,0,0],
           [-1,0,0],
           [0,1,0],
           [0,-1,0],
           [R2, R2, 0],
           [-R2, -R2, 0],
           [R2, -R2, 0],
           [-R2, R2, 0]]
ACTIONS = [np.array(act) for act in ACTIONS]


def closest(action):
    dists = []
    for act in ACTIONS:
        dists.append((act-action).dot(act-action))
    closest = sorted(range(len(dists)), key=lambda i: dists[i])
    return ACTIONS[closest[0]]


# Advance state
def sim(state, action):
    return S.dot(state) + A.dot(closest(action))


# Pos reward
def pos_reward(state):
    return -(state-OBJECTIVE).dot(state-OBJECTIVE)



# Bounds reward
def bnd_reward(state, top=TOP, bot=BOT, pen=PENALTY):
    out = (state > top) + (state < bot)
    if (out.dot(out) > 0):
        return -pen
    return 0


# Obstacle reward
def obs_reward(state, locs, rads, pen=PENALTY):
    for i in range(len(locs)):
        dist = (state-locs[i]).dot(state-locs[i]) 
        if(dist < rads[i]**2):
            return -pen
    return 0


# Reward
def reward(state, locs=[], rads=[], top=TOP, bot=BOT, pen=PENALTY):
    return (pos_reward(state) + bnd_reward(state, top, bot, pen) 
            + obs_reward(state, locs, rads, pen))
    




def features(state, locs=[], rads=[]):
    objective = [state - OBJECTIVE]
    #obstacles = [(state - locs[i]) * ((state-locs[i]).dot(state-locs[i]) < 3*rads[i]**2)
     #            for i in range(len(locs))]
    obstacles = [(state - locs[i]) * (((state-locs[i]).dot(state-locs[i]))**(-1))
                for i in range(len(locs))]
    return np.vstack(objective + obstacles).flatten()


def simple_policy(state, parameters, locs=[], rads=[], 
                  feature_map=features):
    return feature_map(state, locs, rads).dot(parameters)





# Value function
def value(state, params, locs=[], rads=[], 
          policy=simple_policy, discount=0.9, duration=30):
    s = state
    val = 0
    for i in range(duration):
        val += (discount**i) * reward(s, locs, rads)
        s = sim(s, policy(state, params, locs, rads))
    return val





def best_fit(X, Y):
    return np.linalg.pinv(X).dot(Y)


def gen_obstacles(num=3, radius_mean=5, radius_std=2):
    locs = np.random.randint(15, 85, size=(num,2))
    locs = np.hstack((locs, np.zeros(shape=(num,1))))
    rads = np.random.normal(radius_mean, radius_std, size=(num))
    return locs, rads


def learn(iters=5, samples=1000, param=None):
    locs, rads = gen_obstacles()
    n = len(features(START, locs, rads))
    if (param == None):
        param = np.random.normal(0, 1, size=(n, 3))
    
    for t in range(iters):
        states = np.random.uniform(-50, 150, size=(samples,2))
        states = np.hstack((states, np.zeros(shape=(samples,1))))  # Sample states
        
        Y = []
        X = []
        for i in range(samples):
            locs, rads = gen_obstacles()
            
            best_act = ACTIONS[0]
            best_val = value(sim(states[i], ACTIONS[0]), param, locs, rads)
            for act in ACTIONS:
                v = value(sim(states[i], act), param, locs, rads)
                if (v > best_val):
                    best_act = act
                    best_val = v
            Y.append(best_act)
            X.append(features(states[i], locs, rads))
            
        Y = np.vstack(Y)
        X = np.vstack(X)
            
        param = best_fit(X, Y)        
        
        if (t%5 == 0):
            print(value(START, param, locs, rads))
            
    print('Done')
    return param
    

def test(rule):
    locs, rads = gen_obstacles()
    s = START
    
    data = [s]
    for t in range(500):
        s = sim(s, simple_policy(s, rule, locs, rads))
        data.append(s)
    data = np.array(data)


    # (12500) = 25 
    # (12500/25) = 1  # ratio for figsize 8
    size = 8
    plt.figure(figsize=(size,size))
    plt.ylim(-50, 150)
    plt.xlim(-50,150)
    ratio = 12000 / (25**2)

    plt.scatter(data[:,0], data[:,1], marker='.')
    plt.scatter(locs[:,0], locs[:,1], s=(rads*rads)*ratio, color='r')
    plt.grid()
    #plt.scatter([-50], [-50], s=ratio*(25**2))
    
    return locs, rads



rule = learn()
print(test(rule))



