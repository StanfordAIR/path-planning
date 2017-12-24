# -*- coding: utf-8 -*-
"""planner.py

Description

Todo:
    * everything
"""
from typing import Dict

import numpy as np
from geopy.distance import distance as distance_ll

class Environment:
    def __init__(self, boundary: np.ndarray, static_obs: np.ndarray):
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
                       
        # set all fields that are constant between missions
        self.boundary_ll = boundary
        self.min_ll = np.amin(self.boundary_ll, axis=1)[,np.newaxis] # keep shape
        self.max_ll = np.amax(self.boundary_ll, axis=1)[,np.newaxis]
        self.lat_in_ft, self.lon_in_ft = self.ll_in_ft()
        self.boundary_ft = self.ll_to_ft(self.boundary, copy=True)

        # set all fields that are constant within a mission
        self.static_obs_ll = static_obs
        self.static_obs_ft = self.ll_to_ft(self.static_obs_ll, copy=True) 
        #self.graph = #TODO call function to build graph without moving obstacles

    def ll_in_ft(self):
        min_lat, min_lon = (self.min_ll[0,0], self.min_ll[0,0])
        max_lat, max_lon = (self.max_ll[0,0], self.max_ll[1,0])
        lat_size_ll = max_lat - min_lat
        lon_size_ll = max_lon - min_lon
        lat_size_ft = distance_ll((min_lat, min_lon), (max_lat, min_lon)).ft
        lon_size_ft = distance_ll((min_lat, min_lon), (min_lat, max_lon)).ft
        return (lat_size_ft / lat_size_ll, lon_size_ft / lon_size_ll)

    def ll_to_ft(self, points: np.ndarray, copy=False):
        if copy:
            modified_points = np.copy(points)
        else:
            modified_points = points #TODO check that this doesn't copy
        modified_points[0:2] -= self.min_ll
        modified_points[0] *= self.lat_in_ft
        modified_points[1] *= self.lon_in_ft
        return modified_points


class Planner:
    DEFAULT_PARAMS = {"param1": None}

    def __init__(self, environment: Environment,
                       params: Dict[str, Any] = None): # algorithm parameters
        if params:
            self.params = params
        else:
            self.params = DEFAULT_PARAMS

