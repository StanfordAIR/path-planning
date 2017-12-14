# -*- coding: utf-8 -*-
"""
learner.py
Contains the learning agent class for different policy iteration models.
"""
import numpy as np
import numpy.random as nprand

import policy_utils as ut


class PolicyAgent():
    def __init__(self, policy, update, feature, label, params):
        # Custom functions for the model.
        self.policy = policy
        self.update = update
        
        self.feature = feature
        self.label = label
        
        self.params = params
        self.history = []
        self.performances = []
        
        
    def train(self, iters=10, train_sample=500, test_sample=250, 
              feedback=True):
        
        policy, update = self.policy, self.update
        feature, label = self.feature, self.label
        params = self.params
        
        if (feedback):
            print('Starting')
        
        self.history += [params]        
        self.performances += [ut.batch_test(test_sample, policy, params)]
        
        for t in range(iters):
            train_states = ut.gen_training_states(train_sample)
                    
            Y = []
            X = []
            for pos, locs, rads in train_states:
                best_act = ut.ACTIONS[nprand.randint(0, ut.NUM_ACTIONS)]
                best_val = ut.value(ut.sim(pos, best_act), locs, rads, policy, params)
                for a in ut.ACTIONS:
                    v = ut.value(ut.sim(pos, a), locs, rads, policy, params)
                    if (v > best_val):
                        best_act = a
                        best_val = v
                
                Y.append(label(best_act))
                X.append(feature(pos, locs, rads))
                
            params = update(params, X, Y)        
            
            perf = ut.batch_test(test_sample, policy, params)
            self.history += [params[:]]
            self.performances += [perf]
            
            if (feedback):
                print('Success rate:', perf)

            
        if (feedback):
            print('Done')
    