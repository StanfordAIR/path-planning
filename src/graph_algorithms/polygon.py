from typing import List

import numpy as np

from utility_classes import Location

class Polygon:
    """A class that represents the flight boundary and can check point containment
    See http://alienryderflex.com/polygon/ for details.
    """

    def __init__(self, boundary: List[Location]):
        """Precalculates values for checking polygon point-inclusion.
        Args:
            boundary: a list of boundary point locations
        """
        self.corner_count = len(boundary)
        self.poly_x = np.array([p.lon for p in boundary])
        self.poly_y = np.array([p.lat for p in boundary])
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

    def __contains__(self, point: Location) -> bool:
        """Checks whether a point is in the polygon
        Args:
            point: a location
        Returns:
            odd_nodes: true iff point in polygon
        """

        x = point.lon
        y = point.lat
        j = self.corner_count - 1
        odd_nodes = False

        for i in range(self.corner_count):
            if ((self.poly_y[i] < y and self.poly_y[j] >= y)
                or (self.poly_y[j] < y and self.poly_y[i] >= y)):
                odd_nodes ^= (y * self.multiple[i] + self.constant[i] < x)
            j = i

        return odd_nodes


# test if run as main
if __name__ == "__main__":
    boundary = [Location(0.0, 0.0), Location(1.1, 0.0), Location(1.1, 1.1)]
    boundary_poly = Polygon(boundary)
    point = Location(0.5, 1.2)
    print(point in boundary_poly)
