# -*- coding: utf-8 -*-
"""polygon.py

Description

Todo:
    * Add todo
"""
import numpy as np

class Polygon:
    """A class that represents the flight boundary and can check point containment
    See http://alienryderflex.com/polygon/ for details.
    """

    def __init__(self, boundary: np.ndarray):
        """Precalculates values for checking polygon point-inclusion.
        Args:
            boundary: a list of boundary point locations
        """
        self.corner_count = boundary.shape[1]
        self.poly_x = boundary[1]
        self.poly_y = boundary[0]
        self.constant = np.zeros(self.poly_x.size)
        self.multiple = np.zeros(self.poly_x.size)

        j = self.corner_count - 1

        for i in range(self.corner_count):
            if self.poly_y[i] == self.poly_y[j]:
                self.constant[i] = self.poly_x[i]
                self.multiple[i] = 0
            else:
                self.constant[i] = (self.poly_x[i]
                                    - (self.poly_y[i] * self.poly_x[j])
                                       / (self.poly_y[j] - self.poly_y[i])
                                    + (self.poly_y[i] * self.poly_x[i])
                                       / (self.poly_y[j] - self.poly_y[i]))
                self.multiple[i] = ((self.poly_x[j] - self.poly_x[i])
                                    / (self.poly_y[j] - self.poly_y[i]))
            j = i

    def __contains__(self, ll) -> bool:
        """Checks whether a point is in the polygon
        Args:
            point: a location
        Returns:
            odd_nodes: true iff point in polygon
        """
        x = ll[1]
        y = ll[0]
        j = self.corner_count - 1
        odd_nodes = False

        for i in range(self.corner_count):
            if ((self.poly_y[i] < y and self.poly_y[j] >= y)
                or (self.poly_y[j] < y and self.poly_y[i] >= y)):
                odd_nodes ^= (y * self.multiple[i] + self.constant[i] < x)
            j = i

        return odd_nodes
