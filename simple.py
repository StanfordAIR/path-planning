# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:00:16 2017
"""
import numpy as np
import matplotlib.pyplot as plt

# Constants
S = np.array([[1,0,0,1,0,0],
             [0,1,0,0,1,0],
             [0,0,1,0,0,1],
             [0,0,0,0,0,0],
             [0,0,0,0,0,0],
             [0,0,0,0,0,0]])

I = np.array([[0.5,0,0],
             [0,0.5,0],
             [0,0,0.5],
             [1,0,0],
             [0,1,0],
             [0,0,1]])

START = np.zeros(6)
OBJECTIVE = np.array([100,100,0,0,0,0])

R = np.random.randint(-3, 3, (30, len(START)))



def features2(state):
    features = [state-OBJECTIVE]
    return np.hstack(features)


def lin_features(state):
    return np.hstack((state, 1))


class Sim():
    # Policy should be a 3xfeatures matrix.
    def __init__(self, policy, start=np.zeros(6), 
                 features=lin_features):
        self.state = start  # pos, vel
        self.t = 0
        self.policy = policy
        self.features = features
        
    # Return the action given by the policy and current state.
    # TODO Limit acceleration
    def action(self):
        return self.policy.dot(self.features(self.state))
                     
    # Return the next state.
    def next_state(self):
        a = self.action() / np.linalg.norm(self.action())
        return S.dot(self.state) + I.dot(a)
    
    # Advance state.
    def advance(self):
        self.state = self.next_state()
        
       
        

# Reward function.
def reward(state):
    return -(state-OBJECTIVE).dot(state-OBJECTIVE)

# Value function.
def value(policy, state, features=lin_features, time_weight=0.9, iterations=30):
    sim = Sim(policy, state, features)
    val = 0
    for i in range(iterations):
        val += reward(sim.state) * (time_weight**i)
        sim.advance()
    return val    


def psuedo_grad(policy, state, features=lin_features):
    m, n = policy.shape
    delta = np.random.normal(size=(m,n))
    
    change = value(policy+delta, state, features) - value(policy, state, features)
    
    if (change > 0):
        return delta
    else:
        return 0 * delta
    

def main():
    policy = np.random.normal(size=(3,len(features2(START))))
    print(value(policy, START, features2), '\n')
    ITERATIONS = 5000

    
    for i in range(ITERATIONS):        
        # Stochastic gradient ascent (sorta)
        policy += psuedo_grad(policy, np.random.randint(-50, 150, size=(6)), features2)
        
        
        if (i%1000 == 0):
            print(value(policy, START, features2), end=' ')
            
    print('\n\n', value(policy, START, features2))
    
    sim = Sim(policy, START, features2)
    data = []
    for i in range(200):
        data.append(sim.state[:2])
        sim.advance()
        
    plt.figure(figsize=(8,8))
    data = np.array(data)
    plt.plot(data[:,0], data[:,1], '.')
    
    print(policy)
    
    
main()