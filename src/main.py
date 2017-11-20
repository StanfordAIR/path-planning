from graph_algorithms import initial_graph
from graph_algorithms import graph_path
from graph_algorithms.utility_classes import Location, Obstacle

import networkx as nx
import matplotlib.pyplot as plt

boundary = [Location(-50.0, -50.0), Location(-50.0, 100.0),
            Location(150.0, -50.0), Location(150.0, 100.0)]
stat_obstacles = [Obstacle(Location(0.0, 0.0), 20),
                  Obstacle(Location(30.0, 30.0), 10)]
granularity = 7.0
waypoints = [Location(-50.0, -50.0), Location(100.0, 100.0)]

initial_graph, origin = initial_graph.build(boundary, stat_obstacles, granularity)
path, rn_path = graph_path.plan(waypoints, initial_graph, origin, granularity)
print(path)
plt.figure()
pos = nx.get_node_attributes(initial_graph, 'pos')
nx.draw(initial_graph, pos, nodelist=path, font_weight='bold')

plt.figure()
quantization_distance = 1.5 
quantized_path = graph_path.quantize(path, rn_path, quantization_distance)
print(quantized_path.shape)
print(quantized_path)
plt.plot(quantized_path[0], quantized_path[1], 'ro')
plt.show()
