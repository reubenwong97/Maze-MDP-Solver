from env.maze import Maze
from learner.value_iteration import ValueIterationLearner
from learner.policy_iteration import PolicyIterationLearner
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--difficulty', default='default', type=str)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--vtheta', default=0.01, type=float)
parser.add_argument('--ptheta', default=0.01, type=float)

args = parser.parse_args()

env = Maze(args.difficulty)

learner = ValueIterationLearner(env=env)
learner.value_iteration(args.gamma, args.vtheta)
learner.plot_value(save_path='visualisations/value_iteration_utilities_{}.png'.format(env.difficulty))
learner.visualise_policy(save_path='visualisations/value_iteration_policy_{}.png'.format(env.difficulty))
learner.save_value_history()
learner.plot_histories()

learner = PolicyIterationLearner(env=env)
learner.policy_iteration(args.gamma, args.ptheta)
learner.plot_value(save_path='visualisations/policy_iteration_utilities_{}.png'.format(env.difficulty))
learner.visualise_policy(save_path='visualisations/policy_iteration_policy_{}.png'.format(env.difficulty))
learner.save_value_history()
learner.plot_histories()