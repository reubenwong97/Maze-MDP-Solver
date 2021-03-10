from env.maze import Maze
from learner.value_iteration import ValueIterationLearner
from learner.policy_iteration import PolicyIterationLearner
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--difficulty', default='default')
parser.add_argument('--gamma', default=0.99)
parser.add_argument('--theta', default=0.01)

args = parser.parse_args()

env = Maze(args.difficulty)

learner = ValueIterationLearner(env=env)
learner.value_iteration(args.gamma, args.theta)
learner.plot_value(save_path='visualisations/value_iteration_values_{}.png'.format(env.difficulty))
learner.visualise_policy(save_path='visualisations/value_iteration_policy_{}.png'.format(env.difficulty))

learner = PolicyIterationLearner(env=env)
learner.policy_iteration(args.gamma, args.theta)
learner.plot_value(save_path='visualisations/policy_iteration_utilities_{}.png'.format(env.difficulty))
learner.visualise_policy(save_path='visualisations/policy_iteration_policy_{}.png'.format(env.difficulty))
