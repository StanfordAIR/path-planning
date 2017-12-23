import random
import pickle
import numpy as np
import numpy.linalg as lin
from sklearn.neural_network import MLPRegressor
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
PENALTY = 10**8

START = np.zeros(3)
OBJECTIVE = np.array([100,100,0])

R2 = np.sqrt(2)/2
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
    return -np.sqrt((state-OBJECTIVE).dot(state-OBJECTIVE))

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

def obs_reward_tol(state, locs, rads, tol):
    for i in range(len(locs)):
        dist = (state-locs[i]).dot(state-locs[i])
        if(dist < (rads[i]-tol)**2):
            return -1
    return 0


# Reward
def reward(state, locs=[], rads=[], top=TOP, bot=BOT, pen=PENALTY):
    return (pos_reward(state) + bnd_reward(state, top, bot, pen)
            + obs_reward(state, locs, rads, pen))

def features(state, locs=[], rads=[]):
    srt = sorted(range(len(locs)), key=lambda i: lin.norm(state-locs[i]) - rads[i])
    objective = [state - OBJECTIVE]
    obstacles = [(state - locs[srt[i]]) * (lin.norm(state-locs[srt[i]])**2 < 3*rads[srt[i]]**2)
                for i in range(len(srt))]
    return np.vstack(objective + obstacles).flatten()

def simple_policy(state, parameters, locs=[], rads=[],
                  feature_map=features):
    return feature_map(state, locs, rads).dot(parameters)

def nn_policy(state, model, locs=[], rads=[],
              feature_map=features, to_print=False):
    X = feature_map(state,locs,rads)
    X = X.reshape((1,len(X)))
    y = model.predict(X)
    return y[0]


def verify_state(s, locs, rads):
    while True:
        if obs_reward(s, locs, rads) < 0:
            s = np.hstack( (np.random.uniform(-50,150, size=2), np.zeros(shape=1)))
        else: return s
        
# Value function
def value(state, params, locs=[], rads=[],
          policy=simple_policy, discount=0.9, duration=30):
    s = state
    val = 0
    for i in range(duration):
        val += (discount**i) * reward(s, locs, rads)
        s = sim(s, policy(s, params, locs, rads))
    return val

def best_fit(X, Y):
    return np.linalg.pinv(X).dot(Y)

def is_valid(loc, rad):
    a = np.sqrt((START - loc).dot(START-loc)) > rad
    b = np.sqrt((OBJECTIVE - loc).dot(OBJECTIVE - loc)) > rad
    return a and b

def gen_obstacles(num=5, radius_mean=15, radius_std=5):
    locs = np.random.randint(-50, 150, size=(num,2))
    locs = np.hstack((locs, np.zeros(shape=(num,1))))
    rads = np.random.normal(radius_mean, radius_std, size=(num))
    good_locs = []
    good_rads = []
    for i in range(num):
        if is_valid(locs[i], rads[i]):
                good_locs.append(locs[i])
                good_rads.append(rads[i])
        else:
            good_locs.append(np.array([-50,-50,0]))
            good_rads.append(0)
            
    return np.array(good_locs), np.array(good_rads)

def init_model(units):
    model = MLPRegressor(hidden_layer_sizes=units,
                         activation='relu',
                         solver='adam',
                         learning_rate_init=0.001,
                         tol=0.00001,
                         shuffle=False)
    return model

def learn(iters=6, samples=1000, units=(100), param=None):   
    locs, rads = gen_obstacles()
    n = len(features(START, locs, rads))
    if (param == None):
        param = np.random.normal(0, 1, size=(n, 3))
        
    nn_model = init_model(units)
    model_trained = False
    data = []
    for t in range(iters):
        print(t, ": ", end='')
        states = np.random.uniform(-50, 150, size=(samples,2))
        states = np.hstack((states, np.zeros(shape=(samples,1))))  # Sample states

        Y = []
        X = []
        print("Simulating",end='')
        for i in range(samples):
            locs, rads = gen_obstacles()
            s_i = verify_state(states[i], locs, rads)
            
            model = nn_model if model_trained else param
            policy = nn_policy if model_trained else simple_policy

            best_act = random.choice(ACTIONS)
            best_val = value(sim(s_i, best_act), model, locs, rads, policy)

            for act in ACTIONS:
                v = value(sim(s_i, act), model, locs, rads, policy)
                if (v > best_val):
                    best_act = act
                    best_val = v
            
            if i % 100 == 0:
                print('. ', end='')
            Y.append(best_act)
            X.append(features(states[i], locs, rads))

        print("Training... ", end='')

        Y = np.vstack(Y)
        X = np.vstack(X)

        param = best_fit(X, Y)
        nn_model.fit(X, Y)
        model_trained = True

        print("Trained. Score:", nn_model.score(X,Y))
        rate, baseline, std_r, std_b = full_test(nn_model, (10,50), 5)
        print("Success Rate:", rate, '|', baseline, '|', std_r, std_b)
        data.append((rate, baseline, std_r, std_b))
            
        if rate > 0.7:
            filename = '_'.join([str(t), str(rate)[2:], str(len(units)), "nnfpi"])
            pickle.dump(nn_model, open(filename, 'wb'))
            print("Saved.")
        
    print('Done')
    return nn_model, data

def circle(loc, rad, resolution=500):
    """
    Generate the x,y coordinates that define a circle.
    """
    t = np.linspace(0, 2*np.pi, resolution)
    return rad * np.cos(t) + loc[0] , rad * np.sin(t) + loc[1]

def dist(a, b):
    return np.sqrt((a - b).dot(a - b))

def single_test(model, tol, locs, rads):
    s = START
    for t in range(500):
        s = sim(s, nn_policy(s, model, locs, rads))
        if dist(s, OBJECTIVE) < tol:
            return 1
        if obs_reward_tol(s, locs, rads, 1) < 0:
            return 0
    return 0

def base_test(tol, locs, rads):
    s = START
    for t in range(500):
        s = sim(s, ACTIONS[4])
        if dist(s, OBJECTIVE) < tol:
            return 1
        if obs_reward_tol(s, locs, rads, 1) < 0:
            return 0
    return 0
            
def test_success_rate(model, n, tol):
    success = 0. 
    baseline = 0.
    for i in range(n):
        locs, rads = gen_obstacles()
        success += single_test(model, tol, locs, rads)
        baseline += base_test(tol, locs, rads)
    return success / n, baseline / n

def full_test(model, test, tol):
    rates = []
    baselines = []
    for i in range(test[0]):
        r, b = test_success_rate(model, test[1], tol)
        rates.append(r)
        baselines.append(b)
    return np.mean(rates), np.mean(baselines), np.std(rates), np.std(baselines)

def test(rule, locs=None, rads=None, visual=False):
    if (locs == None):
        locs, rads = gen_obstacles()
    
    s = START
    data = [s]
    for t in range(500):
        s_next = sim(s, nn_policy(s, rule, locs, rads))
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

def plot_learning_curve(data1, data2, err=False):
    epochs = np.arange(len(data1))
    if err:
        plt.errorbar(epochs, data1[:,0], yerr=data[:,2])
        plt.errorbar(epochs, data1[:,1], yerr=data[:,3])
    else: 
        plt.plot(epochs, data1[:,0])
        plt.plot(epochs, data1[:,1])
    
    plt.plot(epochs, data1[:,0:20])
    plt.xlabel("Epochs")
    plt.ylabel("Sample Success Rate")
    plt.legend(["Deep FPI", "Linear FPI", "Baseline"])
    plt.show()