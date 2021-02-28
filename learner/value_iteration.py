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

    def _to_state(self, location):
        return location[0] + location[1]

    def _to_location(self, state):
        row = state // self.env.shape[0]
        col = state - row * self.env.shape[1]

        return (row, col)

    def bellman_optimality_update(self, s, gamma):
        Q_values = []
        location = self._to_location(s)
        for a in self.env.actions:
            q_value = 0
            for next_location, probability in self.env.transitions(location, a):
                next_state = self._to_state(next_location)
                reward = self.env.get_reward(next_location)
                q_value += probability * (reward + gamma * self.V[next_state])
            Q_values.append(q_value)
        self.V[s] = max(Q_values)

    #! investigate better ways of looping through
    def value_iteration(self, gamma, theta):
        while True:
            delta = 0 # for stopping threshold theta
            for i in range(self.env.shape[0]):
                for j in range(self.env.shape[1]):
                    tile = self.env.maze[i, j]
                    location = tile.location
                    s = self._to_state(location)
                    print('Values of i, j:', i, j)
                    print('At location:', location)
                    print('Updating state:', s)
                    v = self.V[s]
                    self.bellman_optimality_update(s, gamma)
                    delta = max(delta, abs(v - self.V[s]))
            if delta < theta:
                break     
        for i in range(self.env.shape[0]):
            for j in range(self.env.shape[1]):  
                tile = self.env.maze[i, j]
                location = tile.location     
                s = self._to_state(location)
                self.q_greedify_policy(s, gamma)     

    def q_greedify_policy(self, s, gamma):
        Q_values = []
        location = self._to_location(s)
        for a in self.env.actions:
            q_value = 0
            for next_location, probability in self.env.transitions(location, a):
                next_state = self._to_state(next_location)
                reward = self.env.get_reward(next_location)
                q_value += probability * (reward + gamma * self.V[next_state])
            Q_values.append(q_value)
        one_hot = np.zeros(self.env.n_actions)
        one_hot[np.argmax(Q_values)] = 1.0
        self.pi[s] = one_hot