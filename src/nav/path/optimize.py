# -*- coding: utf-8 -*-
"""optimize.py
Smooths the quantized graph path
Todo:
    * Write
"""
from nav.utility.classes import Location
from nav.utility.classes import Obstacle

from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg

def optimize_path(quantized_path: np.ndarray, boundary: List[Location],
                  stat_obstacles: List[Obstacle], params, verbose=True) -> np.ndarray:
    """Smooths the graph path
    Args:
        quantized_path: TODO
        boundary: flight boundary points giving non-convex polygon
        stat_obstacles: the locations and sizes of the stationary obstacles
    Returns:
        flight_path: TODO
    """
    
    # Parameters
    MAX_ITER = params["optim_max_iter"]
    INIT_STEP_SIZE = params["optim_init_step_size"]
    MIN_STEP = params["optim_min_step"]
    RESET_STEP_SIZE = params["optim_reset_step_size"]
    COOLING_SCHEDULE = params["optim_cooling_schedule"]
    FAST_COOLING_SCHEDULE = params["optim_fast_cooling_schedule"]
    INIT_CONSTRAINT_HARDNESS = params["optim_init_constraint_hardness"]
    INIT_SPRING_HARDNESS = params["optim_init_spring_hardness"]
    MAX_TIME_INCREASE = params["optim_max_time_increase"]
    INIT_MOMENTUM = params["optim_init_momentum"]
    MOMENTUM_CHANGE = params["optim_momentum_change"]
    SCALE = params["optim_scale"]
    
    print("quantized_path.shape = {}".format(quantized_path.shape))
    num_points = quantized_path.shape[1]
    print("num_points = {}".format(num_points))
    points = quantized_path * SCALE

    curr_pos = np.copy(points)
    constraint_hardness = INIT_CONSTRAINT_HARDNESS
    
    obstacle_centers = [[obs.location.lat, obs.location.lon] for obs in stat_obstacles]
    obstacle_radii = [obs.radius for obs in stat_obstacles]
    circ_pos = np.array(obstacle_centers).T * SCALE
    circ_radius2 = (np.array(obstacle_radii) * SCALE) ** 2
    
    count = 0
    curr_step = INIT_STEP_SIZE
    prev_loss = float('inf')
    momentum = INIT_MOMENTUM
    
    prev_velocity = np.zeros(curr_pos.shape)
    curr_velocity = np.copy(prev_velocity)
    
    times_increased = 0
    converged = False
    
    for i in range(MAX_ITER+1):
        while True:
            # Accelerated gradient (e.g. with momentum)
            curr_velocity = momentum * prev_velocity \
                            - curr_step * complete_dloss(curr_pos, circ_pos, circ_radius2,
                                                         constraint_hardness,
                                                         INIT_SPRING_HARDNESS)
            proposed_pos = curr_pos + curr_velocity
            proposed_pos[0][0] = 0.0
            proposed_pos[1][0] = 0.0
            proposed_pos[0][-1] = 100.0 * SCALE
            proposed_pos[1][-1] = 100.0 * SCALE
            curr_loss = complete_loss(proposed_pos, circ_pos, circ_radius2,
                                      constraint_hardness, INIT_SPRING_HARDNESS)
            
            # Ensure decrease
            if curr_loss < prev_loss:
                curr_pos = proposed_pos
                prev_velocity = curr_velocity
                prev_loss = curr_loss
                curr_step *= 2
                times_increased = 0
                break
            # Otherwise, decrease the step size
            else:
                curr_step *= .8
            if curr_step < MIN_STEP:
                if times_increased > MAX_TIME_INCREASE:
                    if verbose:
                        print('Cooling schedule is finished. Converged to final result.')
                    converged = True
                    break
                # Once the cooling step converges, increase the constraint hardness
                # such that the optimization can finish
                if verbose:
                    print('Converged in cooling step; increasing schedule'
                          '(C={}) and resetting velocity'.format(constraint_hardness))
                prev_velocity = np.zeros(prev_velocity.shape)
                constraint_hardness *= FAST_COOLING_SCHEDULE
                
                curr_step = RESET_STEP_SIZE;
                times_increased += 1
    
        if converged:
            break
        constraint_hardness *= COOLING_SCHEDULE
        
        
        if i%10==0 and verbose:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plot_path(curr_pos, circ_pos, np.sqrt(circ_radius2), fig, ax, 'Iteration: {}'.format(i))
            fig.savefig('images/iter{:02d}.png'.format(count))
            plt.close(fig)
            count += 1
            print('On iteration {}'.format(i))
            print('Current step size : {}'.format(curr_step))
       
    
    flight_path = curr_pos / SCALE
    return flight_path

def plot_path(points, circles, radii, fig, ax, plot_title=''):
    ax.plot(points[0,:], points[1,:], '-')
    for idx in range(len(radii)):
        curr_circle = circle(circles[0, idx], circles[1, idx], radii[idx])
        ax.plot(curr_circle[0], curr_circle[1], 'r')
        
    fig.gca().set_aspect('equal')
    ax.set_title(plot_title)
    #ax.set_xlim(-1, 1)
    #ax.set_ylim(-.7, .7)

def circle(x, y, r, n=100):
    t = np.linspace(0, 2*np.pi, n)
    return r*np.cos(t)+x, r*np.sin(t)+y

def phi(x):
    q = np.exp(x)
    return 1/(1+q)

def d_phi(x):
    q = 1/(1+np.exp(x))
    return q*(q-1) #-q*(1-q)

vec_phi = np.vectorize(phi)
vec_d_phi = np.vectorize(d_phi)

def complete_loss(pos_mat, circ_pos, circ_radius2, C, K):
    loss_obstacles = loss(pos_mat, circ_pos, circ_radius2, C)
    loss_diff = sum(linalg.norm(pos_mat[:,1:] - pos_mat[:,:-1], axis=0))
    return loss_obstacles + K*loss_diff

def loss(pos_mat, circ_pos, circ_radius2, C):
    return np.sum(np.apply_along_axis(lambda x: compute_loss(x, circ_pos, circ_radius2, C), 0, pos_mat))

def compute_loss(curr_pos, circ_pos, circ_radius2, C):
    all_dist = C*(linalg.norm(circ_pos - curr_pos[:,np.newaxis], axis=0)**2/circ_radius2 - 1)
    return np.sum(phi(all_dist))

def complete_dloss(pos_mat, circ_pos, circ_radius2, C, K):
    dloss_obstacles = dloss(pos_mat, circ_pos, circ_radius2, C)[:,1:-1]
    diff_mat = -pos_mat[:,:-2] + 2*pos_mat[:,1:-1] - pos_mat[:,2:]
    zeros_mat = np.zeros(2)[:,np.newaxis]
    return np.c_[zeros_mat, dloss_obstacles + K*diff_mat, zeros_mat]

def dloss(pos_mat, circ_pos, circ_radius2, C):
    return np.apply_along_axis(lambda x: compute_dloss(x, circ_pos, circ_radius2, C), 0, pos_mat)

def compute_dloss(curr_pos, circ_pos, circ_radius2, C):
    dif_mat = curr_pos[:,np.newaxis] - circ_pos
    all_dist = C*(linalg.norm(dif_mat, axis=0)**2/circ_radius2 - 1)
    return 2*C*np.sum(d_phi(all_dist)*dif_mat/circ_radius2, axis=1)
