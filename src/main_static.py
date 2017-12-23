from nav.graph.initial import build_graph
from nav.graph.path import plan_graph_path
from nav.graph.path import quantize_graph_path
from nav.path.optimize import optimize_path
from nav.utility.classes import Location, Obstacle

import networkx as nx
import matplotlib.pyplot as plt

# State
boundary = [Location(-50.0, -50.0), Location(-50.0, 150.0),
            Location(150.0, -50.0), Location(150.0, 150.0)]

stat_obstacles = [Obstacle(Location(15.0, -15.0), 10),
                  Obstacle(Location(30.0, 40.0), 20),
                  Obstacle(Location(80.0, 70.0), 15)]

# Task
waypoints = [Location(0.0, 0.0), Location(100.0, 100.0)]

# HyperParameters
granularity = 7.0
quantization_distance = 1.5 

### Script
fig, ax = plt.subplots(1, 3, figsize=(12,4))

# Graph path
initial_graph, origin_loc = build_graph(boundary, stat_obstacles, granularity)
graph_path = plan_graph_path(waypoints, initial_graph, origin_loc, granularity)

pos = nx.get_node_attributes(initial_graph, 'pos')
nx.draw(initial_graph, pos, nodelist=graph_path, font_weight='bold', ax=ax[0])

# Quantized Path
quantized_path = quantize_graph_path(graph_path, origin_loc, granularity, quantization_distance)
ax[1].axis((-50.0,150.0,-50.0,150.0))
ax[1].plot(quantized_path[0], quantized_path[1], '.')
for obs in stat_obstacles:
    circle = plt.Circle((obs.location.lat, obs.location.lon),
                        obs.radius, color='r', fill=False)
    ax[1].add_artist(circle)

# Flight Path
flight_path = optimize_path(quantized_path, boundary, stat_obstacles)
ax[2].axis((-50.0, 150.0, -50.0, 150.0))
ax[2].plot(flight_path[0], flight_path[1], '.')
for obs in stat_obstacles:
    circle = plt.Circle((obs.location.lat, obs.location.lon),
                        obs.radius, color='r', fill=False)
    ax[2].add_artist(circle)

plt.show()
