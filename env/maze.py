import numpy as np
import os
from env.tiles import Tile, TileFactory

class Maze(object):
    def __init__(self, maze_path):
        self.default_path = os.path.join(os.getcwd(), 'env/mazes/basic.txt')
        self.maze_path = maze_path if maze_path != None else self.default_path
        self.maze_template = np.genfromtxt(self.maze_path, delimiter=',', dtype=np.uint8)
        self.shape = self.maze_template.shape
        self.factory = TileFactory()
        self.actions = ["Up", "Down", "Left", "Right"]
        self.action_orthogonals = {"Up": ["Left", "Right"], "Down": ["Left", "Right"],
            "Left": ["Up", "Down"], "Right": ["Up", "Down"]
        }
        self._build_maze()

    def is_out_of_bounds(self, location):
        i, j = location[0], location[1]
        if i < 0 or i >= self.shape[0] or j < 0 or j >= self.shape[1]:
            return True
        return False

    def _build_maze(self):
        self.maze = np.zeros(self.shape, dtype=object)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.maze[i, j] = self.factory.create_tile(map_value=self.maze_template[i, j], location=(i, j))

    def model(self, location, action):
        #! Do I even need this if I am not interacting with environment
        '''Transition model'''
        #! Yea this samples, highly likely I dont want this
        if np.random.random() < 0.8:
            action = action
        else:
            action = np.random.choice(self.action_orthogonals[action])

        next_state = self.get_new_state(action, location)
        return next_state

    def is_learnable_state(self, location):
        tile = self.maze[location]
        return tile.learnable

    def get_new_state(self, action, location):
        #! Maybe all I need is this?
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
        tile = self.maze[candidate_location]
        if not tile.passable:
            location_valid = False

        final_location = candidate_location if location_valid else location
        return final_location