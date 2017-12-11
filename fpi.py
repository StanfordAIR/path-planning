# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 22:00:12 2017
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as lin
import pickle

# Constants
S = np.array([[1,0,0],
             [0,1,0],
             [0,0,1]])

A = np.array([[1,0,0],
             [0,1,0],
             [0,0,1]])

TOP = 150
BOT = -50
PENALTY = -10**8

START = np.zeros(3)
OBJECTIVE = np.array([100,100,0])

R2 = np.sqrt(2) / 2
nACTONS = 8
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
    return state + action


# Pos reward
def pos_reward(state):
    return -lin.norm(state-OBJECTIVE)**2



# Bounds reward
def bnd_reward(state, top=TOP, bot=BOT, pen=PENALTY):
    out = (state > top) + (state < bot)
    if (out.dot(out) > 0):
        return pen
    return 0


# Obstacle reward
def obs_reward(state, locs, rads, pen=PENALTY):
    for i in range(len(locs)):
        dist = lin.norm(state-locs[i])
        if(dist < rads[i]):
            return pen
    return 0


# Reward
def reward(state, locs=[], rads=[], top=TOP, bot=BOT, pen=PENALTY):
    return (pos_reward(state) + bnd_reward(state, top, bot, pen) 
            + obs_reward(state, locs, rads, pen))
    




def features(state, locs=[], rads=[]):
    sorting = sorted(range(len(locs)), key=lambda i: lin.norm(state-locs[i]) - rads[i])
    objective = [state - OBJECTIVE]
    obstacles = [(state - locs[i]) * (lin.norm(state - locs[i])-rads[i])**(-2) 
                * ( lin.norm(state-locs[i]) < 3*rads[i])
                 for i in sorting]
    #obstacles = [(state - locs[i]) * rads[i] * (lin.norm(state-locs[i]))**(-2)
    #           for i in range(len(locs))]
    return np.vstack(objective + obstacles).flatten()


def simple_policy(state, parameters, locs=[], rads=[], 
                  feature_map=features):
    return closest(feature_map(state, locs, rads).dot(parameters))





# Value function
def value(state, params, locs=[], rads=[], 
          policy=simple_policy, discount=0.9, duration=15):
    s = state
    val = 0
    for i in range(duration):
        val += (discount**i) * reward(s, locs, rads)
        s = sim(s, policy(s, params, locs, rads))
    return val



def dist(a, b):
    return np.sqrt((a - b).dot(a - b))




def single_test(model, tol, locs, rads):
    s = START
    for t in range(500):
        s = sim(s, simple_policy(s, model, locs, rads))
        if dist(s, OBJECTIVE) < tol:
            return 1
        if obs_reward(s, locs, rads) < 0:
            return 0
    return 0

def lin_test(tol, locs, rads):
    s = START
    for t in range(500):
        s = sim(s, ACTIONS[4])
        if (dist(s, OBJECTIVE) < tol):
            return 1
        if (obs_reward(s, locs, rads) < 0):
            return 0
    return 0

            
def test_success_rate(model, n, tol):
    success = 0
    lin_success = 0
    for i in range(n):
        locs, rads = gen_obstacles()
        success += single_test(model, tol, locs, rads)
        lin_success += lin_test(tol, locs, rads)
    return success / n, lin_success / n




def best_fit(X, Y):
    return np.linalg.pinv(X).dot(Y)


def gen_obstacles(num=8, radius_mean=15, radius_std=5):
    locs = np.random.randint(-50, 150, size=(num,2))
    locs = np.hstack((locs, np.zeros(shape=(num,1))))
    rads = np.random.normal(radius_mean, radius_std, size=(num))
    for i in range(num):
        if (dist(locs[i], START) <= rads[i] or dist(locs[i], OBJECTIVE) <= rads[i]):
            locs[i,:2] = -50 * np.ones(2)
            rads[i] = 1
    return locs, rads



def verify_state(s, locs, rads):
    while True:
        if obs_reward(s, locs, rads) < 0:
            s = np.hstack( (np.random.uniform(-50,150, size=2), np.zeros(shape=1)))
        else: 
            return s


def learn(iters=1, samples=500, param=None, quickstop=True):
    locs, rads = gen_obstacles()
    n = len(features(START, locs, rads))
    if (param == None):
        param = np.random.normal(0, 1, size=(n, 3))
    
    print('Starting')
    
    performances = [test_success_rate(param, 100, 20)]
    past_fits = [param]
    
    while(iters > 0):
        for t in range(iters):
            states = np.random.uniform(-50, 150, size=(samples,2))
            states = np.hstack((states, np.zeros(shape=(samples,1))))  # Sample states
        
            Y = []
            X = []
            for i in range(samples):
                locs, rads = gen_obstacles()
                si = verify_state(states[i], locs, rads)                 
            
                best_act = ACTIONS[np.random.randint(0, nACTONS)]
                best_val = value(sim(si, best_act), param, locs, rads)
                for act in ACTIONS:
                    v = value(sim(si, act), param, locs, rads)
                    if (v > best_val):
                        best_act = act
                        best_val = v
                        Y.append(best_act)
                        X.append(features(si, locs, rads))
            
            Y = np.vstack(Y)
            X = np.vstack(X)
            
            param = best_fit(X, Y)        
            
            perf = test_success_rate(param, 500, 20)
            #if (perf[0] <= performances[len(performances)-1][0] - 0.05):
            #    param = old_param
            #    print('No positive change')
            #else:
            performances += [perf]   
            past_fits += [param[:]]
            print('Success rate:', perf, round((perf[0]-perf[1])/(1-perf[1]),2) )
                
            if (perf[0] >= 0.75 and quickstop):
                if(input('Save? ' ) == 'y'):
                    pickle.dump((param, performances, past_fits), open('linear' + str(perf[0]) + '.pickle', 'wb'))
                if(input('Stop early? ') == 'y'):  # Early stopping
                    break
            
                
        iters = int(input('How many more trials? '))
            
    print('Done')
    return param, performances, past_fits



def circle(loc, rad, resolution=500):
    """ 
    Generate the x,y coordinates that define a circle.
    """
    t = np.linspace(0, 2*np.pi, resolution)
    return rad * np.cos(t) + loc[0] , rad * np.sin(t) + loc[1]



def test(rule, locs=None, rads=None, visual=False):
    if (locs == None):
        locs, rads = gen_obstacles()
    
    s = START
    data = [s]
    for t in range(1000):
        s_next = sim(s, simple_policy(s, rule, locs, rads))
        data.append(s[:])
        s = s_next
    data = np.array(data)

    if(visual):
        size = 8
        plt.figure(figsize=(size,size))
        plt.ylim(-50, 150)
        plt.xlim(-50,150)

        plt.scatter(data[:,0], data[:,1], marker='.', s=50)
        for i in range(len(locs)):
            x,y = circle(locs[i], rads[i])
            plt.scatter(x,y, marker='.', color='r', s=1)
        plt.grid()
        
    return data


def main():
    rule, results, past = learn(iters=100, samples=1000, quickstop=False)
    pickle.dump((rule, results, past), open('overnight.pickle', 'wb'))

#rule, results, past = learn()
#print(test(rule))

#main()

#rule, results, past = pickle.load(open('overnight.pickle', 'rb'))
#x = [res[0] for res in results]
#y = [res[1] for res in results]
#plt.plot(x)
#plt.plot(y)

