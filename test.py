import numpy as np
import matplotlib.pyplot as plt
from nav.planner import Environment, Planner

# set parameters
boundary = np.genfromtxt("boundary.csv", delimiter=",").T
static_obs = np.genfromtxt("static_obs.csv", delimiter=",").T

env_params = {"granularity": 200}

print(boundary)
print(static_obs)
env = Environment(boundary, static_obs, env_params)
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
env.display(ax)
plt.show()
print(env)
