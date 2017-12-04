# -*- coding: utf-8 -*-
"""problem.py
Defines the problem of point to point navigation. To be loaded before training model.
"""

area = ((-50.0, -50.0), (150.0, 150.0)) # ((x_min, y_min), (x_max, y_max))
waypoints = ((0.0, 0.0), (100.0, 100.0)) # ((x_start, y_start), (x_end, y_end))
max_obstacle_count = 8 # Each environment will have 1 - N obstacles, with equal probability
obstacle_radius_mean = 25 # Mean of obstacle radius
obstacle_radius_variance = 10 # Variance of obstacle radius
