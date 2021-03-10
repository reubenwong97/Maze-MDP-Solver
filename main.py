from env.maze import Maze
from learner.value_iteration import ValueIterationLearner
from learner.policy_iteration import PolicyIterationLearner
import numpy as np

env = Maze('default')

learner = ValueIterationLearner(env=env)
learner.value_iteration(0.99, 0.01)
learner.plot_value(save_path='visualisations/value_iteration_values.png')
learner.visualise_policy(save_path='visualisations/value_iteration_policy.png')

learner = PolicyIterationLearner(env=env)
learner.policy_iteration(0.99, 0.01)
learner.plot_value(save_path='visualisations/policy_iteration_utilities.png')
learner.visualise_policy(save_path='visualisations/policy_iteration_policy.png')