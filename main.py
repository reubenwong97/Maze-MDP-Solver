from env.maze import Maze
from learner.value_iteration import ValueIterationLearner
from learner.policy_iteration import PolicyIterationLearner
import numpy as np

env = Maze('default')
# learner = ValueIterationLearner(env=env)

# learner.value_iteration(0.99, 0.01)
# # print(learner.pi)
# # print(learner.V.reshape(env.shape))
# learner.plot_value(save_path='visualisations/value_iteration_values.png')

# print(learner.recover_simple_policy())
learner = PolicyIterationLearner(env=env)
learner.policy_iteration(0.99, 0.01)
learner.plot_value(save_path='visualisations/policy_iteration_utilities.png')