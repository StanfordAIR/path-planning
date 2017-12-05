import typing
from typing import Tuple
from collections import namedtuple

class Location:
    #LAT_TO_FT = 364169.9438853815 # 1 deg lat in ft at competition
    #LON_TO_FT = 287590.2671258514 # 1 deg lon in ft at competition
    LAT_TO_FT = 1.0
    LON_TO_FT = 1.0

    # instance methods
    def __init__(self, lat: float, lon: float):
        self.lat = lat
        self.lon = lon

    def __iter__(self):
        yield self.lat
        yield self.lon

    def __str__(self):
        return "Location(lat={}, lon={})".format(self.lat, self.lon)

    def __repr__(self):
        return "Location(lat={}, lon={})".format(self.lat, self.lon)

    def to_grid(self, origin, granularity: float):
        return (int((self.lat - origin.lat) * Location.LAT_TO_FT / granularity),
                int((self.lon - origin.lon) * Location.LON_TO_FT / granularity))

    # class methods
    def granularity_diff(loc1, loc2, granularity: float):
        """granularity in ft"""
        return (int((loc1.lat - loc2.lat) * Location.LAT_TO_FT / granularity),
                int((loc1.lon - loc2.lon) * Location.LON_TO_FT / granularity))

    def from_grid(node: Tuple[int, int], origin, granularity: float):
        """granularity in ft"""
        return Location(origin.lat + node[0] * granularity / Location.LAT_TO_FT,
                        origin.lon + node[1] * granularity / Location.LON_TO_FT)


    def distance(loc1, loc2):
        return ((((loc1.lat - loc2.lat) * Location.LAT_TO_FT) ** 2)
                + (((loc1.lon - loc2.lon) * Location.LON_TO_FT) ** 2)) ** (1/2)


class Obstacle:
    # instance methods
    def __init__(self, location: Location, radius: float):
        self.location = location
        self.radius = radius

    def __str__(self):
        return "Obstacle(location={}, radius={})".format(self.location, self.radius)

    def __contains__(self, loc):
        return bool(Location.distance(self.location, loc) <= self.radius)

