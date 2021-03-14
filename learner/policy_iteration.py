import numpy as np
from learner.learner import Learner
from copy import deepcopy
import os
import json
import matplotlib.pyplot as plt

class PolicyIterationLearner(Learner):
    def __init__(self, env):
        super(PolicyIterationLearner, self).__init__(env)
        self.name = "Policy Iteration"
        self.action_dict = {
            "Up": 0,
            "Down": 1,
            "Left": 2,
            "Right": 3
        }
        # init an all up policy for PolicyIteration
        self.pi = np.zeros((np.prod(self.env.shape), self.env.n_actions))
        self.pi[:, 0] = 1
        self.vis_dir = os.path.join(self.vis_dir, 'policy_iteration', self.env.difficulty)

    ######################################### RL Update, not used here ################################
    # def bellman_update(self, s, gamma, V):                                                          #
    #     Q_values = []                                                                               #
    #     action_probs = []                                                                           #
    #     location = self.env._to_location(s)                                                         #
    #     reward = self.env.get_reward(location)                                                      #
    #     for a in self.env.actions:                                                                  #
    #         q_value = 0                                                                             #
    #         action_idx = self.action_dict[a]                                                        #
    #         action_prob = self.pi[s, action_idx]                                                    #
    #         action_probs.append(action_prob)                                                        #
    #         for next_state, probability in self.env.transitions(s, a):                              #
    #             q_value += reward + probability * (gamma * V[next_state])                           #
    #         Q_values.append(q_value)                                                                #
    #     Q_values = np.asarray(Q_values)                                                             #
    #     action_probs = np.asarray(action_probs)                                                     #
    #     self.V[s] = np.dot(Q_values, action_probs)                                                  #
    ###################################################################################################

    def modified_bellman_update(self, s, gamma, V):
        location = self.env._to_location(s)
        reward = self.env.get_reward(location)
        action = self.env.actions[np.argmax(self.pi[s])]
        q_value = 0
        for next_state, probability in self.env.transitions(s, action):
            q_value += probability * (gamma * V[next_state])
        self.V[s] = reward + q_value

    def policy_evaluation(self, gamma, theta, count):
        i = 0
        while True:
            V_old = deepcopy(self.V)
            delta = 0
            # if i % 50 == 0:
            #     print('On iteration', i)
            for s in self.env.learnable_states:
                v = V_old[s] # cache value for comparison between iterations
                self.modified_bellman_update(s, gamma, V_old)
                delta = max(delta, abs(v - self.V[s]))
            self.value_history[count].update({i: self.V.tolist()})
            i += 1
            if delta < theta:
                # print('Converged within theta margin at iteration', i)
                break

    def policy_improvement(self):
        # wrapper around greedification to be idiomatic with terminology
        policy_stable = True
        for s in self.env.states:
            if s in self.env.learnable_states:
                pi_old = deepcopy(self.pi[s])
                q_old = 0
                for next_state, probability in self.env.transitions(s, self.env.actions[np.argmax(pi_old)]):
                    q_old += probability * self.V[next_state]

                Q_values = []
                for a in self.env.actions:
                    q_val = 0
                    for next_state, probability in self.env.transitions(s, a):
                        q_val += probability * self.V[next_state]
                    Q_values.append(q_val)
                max_q, max_a_idx = max(Q_values), np.argmax(Q_values)

                if max_q > q_old:
                    one_hot = np.zeros(self.env.n_actions)
                    one_hot[max_a_idx] = 1
                    self.pi[s] = one_hot

                if not np.array_equal(self.pi[s], pi_old):
                    policy_stable = False
            else:
                self.pi[s] = np.zeros(self.env.n_actions)

        return policy_stable

    def policy_iteration(self, gamma, theta):
        i = 0
        self.gamma = gamma
        self.theta = theta
        policy_stable = False
        while not policy_stable:
            self.value_history[i] = {}
            self.policy_evaluation(gamma, theta, count=i)
            self.plot_value(it=i+1, save_path=os.path.join(self.vis_dir, 'policy_iteration_utilities_evaluation_{}'.format(i+1)))
            policy_stable = self.policy_improvement()
            self.visualise_policy(save_path=os.path.join(self.vis_dir, 'policy_iteration_policy_evaluation_{}'.format(i+1)),
                it=i+1)
            i += 1

    def plot_histories(self, results_path=None):
        save_path = os.path.join(os.getcwd(), 'visualisations', 'policy_iteration', 'policy_iteration_line_plot.png')
        results_path = os.path.join(os.getcwd(), 'results', 'policy_iteration.json') if results_path == None else results_path
        
        with open(results_path, 'r') as file:
            histories = json.load(file)

        fig = plt.figure()
        ax = plt.axes()

        for policy_iteration_count in histories.keys():
            data = []
            eval_label = '_eval_{}'.format(str([policy_iteration_count]))
            eval_iter_data = histories[policy_iteration_count]
            for iter_data in eval_iter_data.values():
                iter_data = np.asarray(iter_data)
                data.append(iter_data)
            data = np.asarray(data)

            # plotting
            ax.plot(data[:, 0], label='(0, 0)'+eval_label)
            ax.plot(data[:, 30], label='(5, 0)'+eval_label)
            ax.plot(data[:, 35], label='(5, 5)'+eval_label)

        ax.legend()
        ax.set_title('Plot of Utilities for Select States against Iterations')
        ax.set_ylabel('Utilities')
        ax.set_xlabel('Iterations')
        fig.savefig(save_path)

        # close plotting
        plt.cla()
        plt.close()

    def save_value_history(self, save_dir=None, name='policy_iteration.json'):
        super().save_value_history(save_dir=save_dir, name=name)