from env.maze import Maze
from learner.value_iteration import ValueIterationLearner
import numpy as np

env = Maze('default')
learner = ValueIterationLearner(env=env)

learner.value_iteration(0.95, 0.01)
print(learner.pi)
print(learner.V)