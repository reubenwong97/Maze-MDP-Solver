import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from PIL import Image, ImageDraw
import os
import json

class Learner(object):
    def __init__(self, env) -> None:
        self.env = env
        # 1D array representing value table
        self.V = np.zeros(np.prod(self.env.shape))
        # representation of initial policy, that follows Uniform random
        # states | Up  |  Down | Left | Right
       # all init| 0.25| 0.25  | 0.25 | 0.25
        self.pi = np.ones((np.prod(self.env.shape), self.env.n_actions)) / self.env.n_actions
        self.vis_dir = os.path.join(os.getcwd(), 'visualisations')
        self.value_history = {}


    def plot_value(self, annot=True, vmin='auto', cmap='YlGnBu', fmt='.2f', it=None, save_path=None):
        V_shaped = deepcopy(self.V.reshape(self.env.shape))
        V_shaped[V_shaped == 0] = np.nan
        if vmin == 'auto':
            print('Auto vmin selected...')
            vmin = np.nanmin(V_shaped)
            print('Auto vmin determined to be', vmin,'...')
        res = sns.heatmap(V_shaped, annot=annot, vmin=vmin, cmap=cmap, fmt=fmt, 
                    linewidths=0.1, linecolor='gray')
        for _, spine in res.spines.items():
            spine.set_visible(True)
        plt.rcParams["axes.titlesize"] = 10
        plt.rcParams['figure.figsize'] = 9, 7
        if not it:
            plt.title("{}, gamma={}, theta={}, difficulty={}".format(self.name, self.gamma, self.theta, 
                self.env.difficulty))
        else:
            plt.title("{}, gamma={}, theta={}, iter={}, difficulty={}".format(self.name, self.gamma, self.theta, 
                it, self.env.difficulty))
        if save_path:
            fig = res.get_figure()
            fig.tight_layout()
            fig.savefig(save_path)
        else:
            plt.show()
        plt.cla()
        plt.close()

    # wrapper for policy annotations
    def visualise_policy(self, fmt='', it=None, save_path=None):
        labels = self.recover_simple_policy()
        if not it:
            self.plot_value(annot=labels, fmt=fmt, save_path=save_path)
        else:
            self.plot_value(annot=labels, fmt=fmt, it=it, save_path=save_path)

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

    def save_value_history(self, save_dir=None, name=None):
        save_path = os.path.join(os.getcwd(), 'results', name)
        with open(save_path, 'w') as file:
            json.dump(self.value_history, file, indent=4)
        print('Histories saved to {}'.format(save_path))