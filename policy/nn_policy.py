# -*- coding: utf-8 -*-
"""
linear_policy.py
Code for training a linear policy function.
"""
import numpy as np
import numpy.linalg as lin
import numpy.random as nprand

import policy_utils as ut
from learner import PolicyAgent
from sklearn.neural_network import MLPRegressor


def policy(pos, locs, rads, model):
    """
    Generate an action given some state.
    """
    X = feature(pos,locs,rads)
    X = X.reshape((1,len(X)))
    return ut.closest(model.predict(X))

def update(model, X_data, Y_data):
    """
    Update the paramaters.
    """
    X = np.vstack(X_data)
    Y = np.vstack(Y_data)
    return model.fit(X,Y)

def feature(state, locs=[], rads=[]):
    srt = sorted(range(len(locs)), key=lambda i: lin.norm(state-locs[i]) - rads[i])
    objective = [state - ut.FINISH]
    obstacles = [(state - locs[i]) * (lin.norm(state-locs[i])-rads[i])**-2
                for i in srt]
    return np.vstack(objective + obstacles).flatten()

def label(act):
    """
    Labels for each action.
    """
    return ut.ACTIONS[act]

def initialize(units):
    model = MLPRegressor(hidden_layer_sizes=units,
                         activation='relu',
                         solver='adam',
                         learning_rate_init=0.001,
                         tol=0.00001,
                         shuffle=False)
    X_init = np.array([feature(nprand.rand(2), locs=nprand.rand(ut.NUM_OBS,2), rads=nprand.rand(ut.NUM_OBS))
                       for i in range(100)])
    Y_init = nprand.rand(100, 2)
    model.fit(X_init, Y_init)
    return model

units = (100,)
agent = PolicyAgent(policy, update, feature, label, initialize(units))
agent.train(iters=30)