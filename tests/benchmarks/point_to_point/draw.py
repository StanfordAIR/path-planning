# -*- coding: utf-8 -*-
"""draw.py
Draws the first 9 environments from environments.py
"""

import math
import matplotlib.pyplot as plt
from .problem import area
from .problem import waypoints

def grid(num_displays, envs, sols=None):
    n = int(math.sqrt(num_displays))
    fig, ax = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, figsize=(7,7), squeeze=False)
    ((x_min, y_min), (x_max, y_max)) = area
    ((wx_min, wy_min), (wx_max, wy_max)) = waypoints

    for i in range(n * n):
        ax_x = i % n
        ax_y = i // n
        ax[ax_x, ax_y].set_xlim(x_min, x_max)
        ax[ax_x, ax_y].set_ylim(y_min, y_max)
        ax[ax_x, ax_y].plot([wx_min, wx_max], [wy_min, wy_max], '--bo') # start and endpoints
        env = envs[i]
        for obs in env:
            circle = plt.Circle((obs[0], obs[1]),
                                obs[2], color='r', fill=False)
            ax[ax_x, ax_y].add_artist(circle)

        if sols != None:
            sol = sols[i]
            ax[ax_x, ax_y].plot(sol[0], sol[1], 'ro')

    plt.show()
