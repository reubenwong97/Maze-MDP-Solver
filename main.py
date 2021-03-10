from env.maze import Maze
from learner.value_iteration import ValueIterationLearner
from learner.policy_iteration import PolicyIterationLearner
import numpy as np
from sacred import Experiment

ex = Experiment('maze_mdp')
@ex.config
def cfg():
    difficulty = 'default'
    gamma = 0.99
    theta = 0.01

@ex.main
def run(difficulty, gamma, theta):
    env = Maze(difficulty)

    learner = ValueIterationLearner(env=env)
    learner.value_iteration(gamma, theta)
    learner.plot_value(save_path='visualisations/value_iteration_values_{}.png'.format(env.difficulty))
    learner.visualise_policy(save_path='visualisations/value_iteration_policy_{}.png'.format(env.difficulty))

    learner = PolicyIterationLearner(env=env)
    learner.policy_iteration(gamma, theta)
    learner.plot_value(save_path='visualisations/policy_iteration_utilities_{}.png'.format(env.difficulty))
    learner.visualise_policy(save_path='visualisations/policy_iteration_policy_{}.png'.format(env.difficulty))

if __name__ == '__main__':
    ex.run_commandline()