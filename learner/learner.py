import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from PIL import Image, ImageDraw
import os

class Learner(object):
    def __init__(self, env) -> None:
        self.env = env
        # 1D array representing value table
        self.V = np.zeros(np.prod(self.env.shape))
        # representation of initial policy, that follows Uniform random
        # states | Up  |  Down | Left | Right
       # all init| 0.25| 0.25  | 0.25 | 0.25
        self.pi = np.ones((np.prod(self.env.shape), self.env.n_actions)) / self.env.n_actions

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

    def plot_value(self, annot=True, vmin=85, cmap='YlGnBu', save_path=None):
        V_shaped = deepcopy(self.V.reshape(self.env.shape))
        V_shaped[V_shaped == 0] = np.nan
        res = sns.heatmap(V_shaped, annot=annot, vmin=vmin, cmap=cmap, fmt='.2f', 
                    linewidths=0.1, linecolor='gray')
        for _, spine in res.spines.items():
            spine.set_visible(True)
        plt.title("{}, gamma={}, theta={}".format(self.name, self.gamma, self.theta))
        if save_path:
            fig = res.get_figure()
            fig.savefig(save_path)
        else:
            plt.show()

    def visualise_policy(self):
        fig, ax = plt.subplots()
        im = ax.imshow(self.V.reshape(-1, self.env.shape[1]))
        for state in self.env.states:
            loc = self.env._to_location(state)

    # def _init_grid(self):
    #     grid = Image.new('RGB', (60*self.env.shape[0], 60*self.env.shape[1]), (255, 255, 255))
    #     for x in range(0, 60*self.env.shape[1], 60):
    #         x0, y0, x1, y1 = x, 0, x, 60*self.env.shape[0]
    #         column = (x0, y0), (x1, y1)
    #         ImageDraw.Draw(grid).line(column, (128, 128, 128), 1)
    #     for y in range(0, 60*self.env.shape[0], 60):
    #         x0, y0, x1, y1 = 0, y, 60*self.env.shape[1], y
    #         row = (x0, y0), (x1, y1)
    #         ImageDraw.Draw(grid).line(row, (128, 128, 128), 1)
    #     return grid

    # def visualise_policy(self):
    #     assets_path = os.path.join(os.getcwd(), 'assets')
    #     grid = []
    #     pi = self.recover_simple_policy()
    #     for s in pi:
    #         image_states = []
    #         for a in s:
    #             image = Image.open(os.path.join(assets_path, a+'.png')).convert('LA')
    #             image_array = np.asarray(image)
    #             print(image_array.shape)
    #             image_states.append(image_array)
    #         grid.append(np.asarray(image_states))
    #     grid = np.asarray(grid).reshape(pi.shape)
    #     plt.imshow(grid)
    #     plt.show()

    def recover_simple_policy(self):
        actions = self.env.actions
        new_pi = []
        for s, action_dist in enumerate(self.pi):
            loc = self.env._to_location(s)
            if self.env.is_learnable_state(loc):
                action_idx = np.argmax(action_dist)
                action = actions[action_idx]
            else:
                action = 'Wall'
            new_pi.append(action)
        
        return np.array(new_pi).reshape(self.env.shape)