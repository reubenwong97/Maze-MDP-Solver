import numpy as np
import os

from numpy.lib.function_base import diff
from env.tiles import TileFactory

class Maze(object):
    def __init__(self, difficulty='default'):
        self.difficulty = difficulty
        self.maze_path = os.path.join(os.getcwd(), 'env/mazes/{}.txt'.format(difficulty))
        self.maze_template = np.genfromtxt(self.maze_path, delimiter=',', dtype=np.uint8)
        self.shape = self.maze_template.shape
        self.factory = TileFactory()
        # states represented [0, 1, ..., np.prod(self.shape)]
        self.states = np.array([x for x in range(np.prod(self.shape))])
        self.actions = ["Up", "Down", "Left", "Right"]
        self.n_actions = len(self.actions)
        self.action_orthogonals = {"Up": ["Left", "Right"], "Down": ["Left", "Right"],
            "Left": ["Up", "Down"], "Right": ["Up", "Down"]
        }
        self._build_maze()

    def _to_state(self, location):
        return location[0] * self.shape[1] + location[1]

    def _to_location(self, state):
        row = state // self.shape[1]
        # col = state - row * self.shape[1]
        col = state % self.shape[1]

        return (row, col)

    def is_out_of_bounds(self, location):
        i, j = location[0], location[1]
        if i < 0 or i >= self.shape[0] or j < 0 or j >= self.shape[1]:
            return True
        return False

    def _build_maze(self):
        self.learnable_states = []
        self.maze = np.zeros(self.shape, dtype=object)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                new_tile = self.factory.create_tile(map_value=self.maze_template[i, j], location=(i, j))
                self.maze[i, j] = new_tile
                if new_tile.learnable:
                    self.learnable_states.append(self._to_state(new_tile.location))

    def sample_state(self, location, action):
        '''For sampling next states | (s,a)'''
        if np.random.random() < 0.8:
            action = action
        else:
            action = np.random.choice(self.action_orthogonals[action])

        next_state = self.get_new_location(action, location)
        return next_state

    def get_reward(self, location):
        tile = self.maze[location]
        return tile.reward

    def is_learnable_state(self, location):
        tile = self.maze[location]
        return tile.learnable

    def transitions(self, state, action):
        # refactored as state to cater to general conventions
        '''return next states and probabilities'''
        # next state and probability if movement corresponds to selected action
        location = self._to_location(state)
        transitions = []
        majority_location = self.get_new_location(action, location)
        majority_probability = 0.8
        majority_transition = [self._to_state(majority_location), majority_probability]
        transitions.append(majority_transition)

        # transitions if movement is orthogonal to selected action
        minority_probability = 0.1
        orthogonal_actions = self.action_orthogonals[action]
        ortho_action_1 = orthogonal_actions[0]
        minority_location_1 = self.get_new_location(ortho_action_1, location)
        transitions.append([self._to_state(minority_location_1), minority_probability])
        ortho_action_2 = orthogonal_actions[1]
        minority_location_2 = self.get_new_location(ortho_action_2, location)
        transitions.append([self._to_state(minority_location_2), minority_probability])

        # see warning: https://numpy.org/doc/1.19/release/1.19.0-notes.html#deprecate-automatic-dtype-object-for-ragged-input
        # dtype must be specified to avoid warning
        return np.array(transitions, dtype=object)

    def get_new_location(self, action, location):
        'Return next location given a confirmed movement (after going through sampled probabilities)'
        i, j = location
        if action == 'Up':
            candidate_location = (i - 1, j)
        elif action == 'Down':
            candidate_location = (i + 1, j)
        elif action == 'Left':
            candidate_location = (i, j - 1)
        elif action == 'Right':
            candidate_location = (i, j + 1)
        else:
            raise ValueError('Unrecognised action given')

        location_valid = True
        if self.is_out_of_bounds(candidate_location):
            location_valid = False
        if location_valid:
            tile = self.maze[candidate_location]
            if not tile.passable:
                location_valid = False

        final_location = candidate_location if location_valid else location
        return final_location