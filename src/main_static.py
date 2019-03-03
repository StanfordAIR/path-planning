from nav.graph import initial as initial_graph
from nav.graph import path as graph_path
from nav.utility.classes import Location, Obstacle

import networkx as nx
import matplotlib.pyplot as plt

boundary = [Location(-50.0, -50.0), Location(-50.0, 150.0),
            Location(150.0, -50.0), Location(150.0, 150.0)]
obstacles = [
    ((15.0, -15.0), 10),
    ((30.0, 40.0), 20),
    ((80.0, 70.0), 15),
]
stat_obstacles = [Obstacle(Location(*o[0]), o[1]) for o in obstacles]
# stat_obstacles = [Obstacle(Location(15.0, -15.0), 10),
#                   Obstacle(Location(30.0, 40.0), 20),
#                   Obstacle(Location(80.0, 70.0), 15)]
granularity = 7.0
waypoints = [Location(0.0, 0.0), Location(100.0, 100.0)]

initial_graph, origin = initial_graph.build(boundary, stat_obstacles, granularity)
path, rn_path = graph_path.plan(waypoints, initial_graph, origin, granularity)
# print(path)
pos = nx.get_node_attributes(initial_graph, 'pos')
nx.draw(initial_graph, pos, nodelist=path, font_weight='bold')

quantization_distance = 1.5 
quantized_path = graph_path.quantize(path, rn_path, quantization_distance)
# print(quantized_path.shape)
# print(quantized_path)

# plotting stuff
plt.figure()
fig, ax = plt.subplots()
plt.axis((-50.0,150.0,-50.0,150.0))
ax.plot(quantized_path[0], quantized_path[1], '.--')
for o in obstacles:
    circle = plt.Circle(o[0], o[1], color='r')
    ax.add_artist(circle)
# circle1 = plt.Circle((15.0, -15.0), 10.0, color='r')
# circle2 = plt.Circle((30.0, 40.0), 20.0, color='r')
# circle3 = plt.Circle((80.0, 70.0), 15.0, color='r')
# ax.add_artist(circle1)
# ax.add_artist(circle2)
# ax.add_artist(circle3)
plt.show()
