import numpy as np

class ValueIterationLearner(object):
    def __init__(self, env):
        self.env = env
        # 1D array representing value table
        self.V = np.zeros(np.prod(self.env.shape))
        # representation of initial policy, that follows Uniform random
        # states | Up  |  Down | Left | Right
       # all init| 0.25| 0.25  | 0.25 | 0.25
        self.pi = np.ones((np.prod(self.env.shape), self.env.n_actions)) / self.env.n_actions

    def bellman_optimality_update(self, s, gamma):
        Q_values = []
        location = self.env._to_location(s)
        for a in self.env.actions:
            q_value = 0
            for next_state, probability in self.env.transitions(s, a):
                next_location = self.env._to_location(next_state)
                reward = self.env.get_reward(next_location)
                q_value += probability * (reward + gamma * self.V[next_state])
            Q_values.append(q_value)
        self.V[s] = max(Q_values)

    #! investigate better ways of looping through
    def value_iteration(self, gamma, theta):
        i = 0
        while True:
            if i % 50 == 0:
                print('On iteration', i)
            delta = 0 # for stopping threshold theta
            for s in self.env.learnable_states:
                v = self.V[s]
                self.bellman_optimality_update(s, gamma)
                delta = max(delta, abs(v - self.V[s]))
            i += 1
            if delta < theta:
                break
        for s in self.env.learnable_states:
            self.q_greedify_policy(s, gamma)    

    def q_greedify_policy(self, s, gamma):
        Q_values = []
        for a in self.env.actions:
            q_value = 0
            for next_state, probability in self.env.transitions(s, a):
                next_location = self.env._to_location(next_state)
                reward = self.env.get_reward(next_location)
                q_value += probability * (reward + gamma * self.V[next_state])
            Q_values.append(q_value)
        one_hot = np.zeros(self.env.n_actions)
        one_hot[np.argmax(Q_values)] = 1.0
        self.pi[s] = one_hot