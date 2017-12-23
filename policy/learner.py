# -*- coding: utf-8 -*-
"""
learner.py
Contains the learning agent class for different policy iteration models.
"""
import numpy as np
import numpy.random as nprand
import matplotlib.pyplot as plt
from copy import deepcopy

import policy.policy_utils as ut


class PolicyAgent():
    def __init__(self, policy, update, feature, label, params,
                 history=[], performances=[]):
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
        """
        Learn a new model using policy iteration.
        """
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
                best_act = nprand.randint(0, ut.NUM_ACTIONS)
                best_val = ut.value(ut.sim(pos, ut.ACTIONS[best_act]), 
                                    locs, rads, policy, params)
                for a in range(ut.NUM_ACTIONS):
                    v = ut.value(ut.sim(pos, ut.ACTIONS[a]), 
                                 locs, rads, policy, params)
                    if (v > best_val):
                        best_act = a
                        best_val = v
                
                Y.append(label(best_act))
                X.append(feature(pos, locs, rads))
                
            params = update(params, X, Y)        
            
            perf = ut.batch_test(test_sample, policy, params)
            self.history += [deepcopy(params)]
            self.performances += [perf]
            
            if (feedback):
                print(t, 'Success rate:', perf[0],
                      '/', perf[1],
                      '\t', round(perf[0]-perf[1],2))
            
        self.params = params
        if (feedback):
            print('Done')

    
    def sample_run(self):
        """
        Test the current parameters and plot performance on a sample map.
        """
        ut.sample_run(self.policy, self.params)
        
    
    def get_path(self, locs, rads):
        """
        Get the path the current model would generate.
        """
        return ut.test(locs, rads, self.policy, self.params)[1]
    
    
    def plot_history(self):
        """
        Plot the training history.
        """
        y1 = [perf[0] for perf in self.performances]
        y2 = [perf[1] for perf in self.performances]
        
        plt.plot(y1)
        plt.plot(y2, label='Baseline')
        plt.legend()
        plt.show()
       
        
    def get_data(self):
        """
        Return the object data for reloading later.
        """
        return (self.params, self.history, self.performances)
        
        
