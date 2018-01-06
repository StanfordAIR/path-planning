# -*- coding: utf-8 -*-
"""planner.py

Description

Todo:
    * everything
"""
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import distance as distance_ll

from nav.graph.graph import FlightGraph

class Environment:
    def __init__(self, boundary: List,
                 static_obs: List,
                 params: Dict[str,Any]):
        """An Environment describes the static elements of the
        flight environment for a mission.

        Args:
            boundary: Array of points describing flight boundary.
                The array is structured [[latitudes],[longitudes]].
            static_obs: Array of static objects.
                The array is structured [[latitudes],[longitudes],[radii in ft]].

        Todo:
            * ...
        """
        # set all parameters
        self.granularity = params['granularity']
                       
        # set all fields that are constant between missions
        self.boundary_ll = np.array(boundary).T
        self.min_ll = np.amin(self.boundary_ll, axis=1)[:,np.newaxis] # keep shape
        self.max_ll = np.amax(self.boundary_ll, axis=1)[:,np.newaxis]
        self.lat_in_ft, self.lon_in_ft = self._ll_in_ft()
        self.boundary_ft = self.ll_to_ft(self.boundary_ll, copy=True)

        # set all fields that are constant within a mission
        self.static_obs_ll = np.array(static_obs).T
        self.static_obs_ft = self.ll_to_ft(self.static_obs_ll, copy=True) 
        self.graph = FlightGraph(self.boundary_ft, self.static_obs_ft,
                                         self.granularity)

    def ll_to_ft(self, points: np.ndarray, copy: bool = False):
        if copy:
            modified_points = np.copy(points)
        else:
            modified_points = points #TODO check that this doesn't copy
        modified_points[0:2] -= self.min_ll
        modified_points[0] *= self.lat_in_ft
        modified_points[1] *= self.lon_in_ft
        return modified_points

    def display(self, ax):
        # Draw Path
        ax.plot(Environment._wrap1(self.boundary_ft[1]),
                Environment._wrap1(self.boundary_ft[0]))

        # Draw Obstacles
        for obs_idx in range(self.static_obs_ft.shape[1]):
            y = self.static_obs_ft[0,obs_idx]
            x = self.static_obs_ft[1,obs_idx]
            r = self.static_obs_ft[2,obs_idx]
            circle = plt.Circle((x, y), r, color='r', zorder=0)
            ax.add_artist(circle)

        # Draw Graph
        nodes_ft = self.graph.base_nodes_ft
        ax.scatter(nodes_ft[1], nodes_ft[0], s=1)

        # set axis labels and limits to match ft and ll
        ax.set_xlabel("feet east")
        ax.set_xlim([np.amin(self.boundary_ft[1]),np.amax(self.boundary_ft[1])])
        ax.set_ylabel("feet north")
        ax.set_ylim([np.amin(self.boundary_ft[0]),np.amax(self.boundary_ft[0])])

        ax_lon = ax.twiny() # second set of axes for lat,lon
        ax_lat = ax.twinx() # second set of axes for lat,lon
        ax_lon.set_xlabel("longitude")
        ax_lon.set_xlim([np.amin(self.boundary_ll[1]),np.amax(self.boundary_ll[1])])
        ax_lat.set_ylabel("latitude")
        ax_lat.set_ylim([np.amin(self.boundary_ll[0]),np.amax(self.boundary_ll[0])])

    def _ll_in_ft(self):
        min_lat, min_lon = (self.min_ll[0,0], self.min_ll[1,0])
        max_lat, max_lon = (self.max_ll[0,0], self.max_ll[1,0])
        lat_size_ll = max_lat - min_lat
        lon_size_ll = max_lon - min_lon
        lat_size_ft = distance_ll((min_lat, min_lon), (max_lat, min_lon)).ft
        lon_size_ft = distance_ll((min_lat, min_lon), (min_lat, max_lon)).ft
        return (lat_size_ft / lat_size_ll, lon_size_ft / lon_size_ll)

    def _wrap1(array):
        return np.append(array, array[0])

