import numpy as np
from learner.learner import Learner
from copy import deepcopy

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

    def bellman_update(self, s, gamma, V):
        Q_values = []
        action_probs = []
        location = self.env._to_location(s)
        reward = self.env.get_reward(location)
        for a in self.env.actions:
            q_value = 0
            action_idx = self.action_dict[a]
            action_prob = self.pi[s, action_idx]
            action_probs.append(action_prob)
            for next_state, probability in self.env.transitions(s, a):
                q_value += reward + probability * (gamma * V[next_state])
            Q_values.append(q_value)
        Q_values = np.asarray(Q_values)
        action_probs = np.asarray(action_probs)
        self.V[s] = np.dot(Q_values, action_probs)

    def policy_evaluation(self, gamma, theta):
        self.gamma = gamma
        self.theta = theta
        i = 0
        while True:
            V_old = deepcopy(self.V)
            delta = 0
            if i % 50 == 0:
                print('On iteration', i)
            for s in self.env.learnable_states:
                v = V_old[s] # cache value for comparison between iterations
                self.bellman_update(s, gamma, V_old)
                delta = max(delta, abs(v - self.V[s]))
            i += 1
            if delta < theta:
                print('Converged within theta margin at iteration', i)
                break

    def policy_improvement(self):
        # wrapper around greedification to be idiomatic with terminology
        for s in self.env.states:
            if s in self.env.learnable_states:
                self.q_greedify_policy(s, self.gamma)
            else:
                self.pi[s] = np.zeros(self.env.n_actions)

    def policy_iteration(self, gamma, theta):
        policy_stable = False
        while not policy_stable:
            self.policy_evaluation(gamma, theta)