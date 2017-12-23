# -*- coding: utf-8 -*-
"""
linear_policy.py
Code for training a linear policy function.
"""
import numpy as np
import numpy.linalg as lin

import policy.policy_utils as ut
from policy.learner import PolicyAgent
import pickle


def policy(pos, locs, rads, params):
    """
    Generate an action given some state.
    """
    return ut.closest(feature(pos, locs, rads).dot(params))


def update(params, X_data, Y_data):
    """
    Update the paramaters.
    """
    X = np.vstack(X_data)
    Y = np.vstack(Y_data)
    return lin.pinv(X).dot(Y)


def feature(pos, locs, rads):
    """
    Map the state to an advanced feature mapping.
    """
    sorting = sorted(range(len(locs)), key=lambda i: lin.norm(pos-locs[i]) - rads[i])
    objective = [pos - ut.FINISH]
    obstacles = [(pos - locs[i]) * (lin.norm(pos - locs[i])-rads[i])**(-2) 
                * ( lin.norm(pos-locs[i]) < 3*rads[i])
                 for i in sorting]
    #obstacles = [(pos - locs[i]) * (lin.norm(pos - locs[i]))**(-2) 
    #            * ( lin.norm(pos-locs[i]) < 3*rads[i])
    #            for i in sorting]
    return np.vstack(objective + obstacles).flatten()


def label(act):
    """
    Labels for each action.
    """
    return ut.ACTIONS[act]


def initialize():
    locs, rads = ut.gen_obstacles()
    n = len(feature(ut.START, locs, rads))
    return np.zeros(n)

if __name__ == "__main__":
    data = pickle.load(open('linear_data2.pkl', 'rb'))
    agent = PolicyAgent(policy, update, feature, label, data[0], data[1], data[2])
    agent.train(iters=70, train_sample=1000)

