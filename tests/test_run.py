from . import context

import numpy as np
import matplotlib.pyplot as plt

import nav.env
import nav.plan

# set parameters
CONFIG_DIR = './tests/config/'
boundary = np.genfromtxt(CONFIG_DIR + 'boundary.csv', delimiter=',').T
static_obs = np.genfromtxt(CONFIG_DIR + 'static_obs.csv', delimiter=',').T

env_params = {"granularity": 100}
planner_params = {}

env = nav.env.Environment(boundary, static_obs, env_params)
planner = nav.plan.Planner(env, planner_params)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
env.display(ax)
plt.show()
print(env)
