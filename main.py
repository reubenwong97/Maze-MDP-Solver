from env.maze import Maze
from learner.value_iteration import ValueIterationLearner
import numpy as np

env = Maze('default')
learner = ValueIterationLearner(env=env)

learner.value_iteration(0.99, 0.01)
# print(learner.pi)
print(learner.V.reshape(env.shape))
learner.plot_value(save_path='visualisations/value_iteration_values.png')

print(learner.recover_simple_policy())
learner.visualise_policy()