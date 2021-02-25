import numpy as np
import os

class Maze(object):
    def __init__(self, maze_path):
        self.default_path = os.path.join(os.getcwd(), 'env/mazes/basic.txt')
        self.maze_path = maze_path if maze_path != None else self.default_path
        self.maze = np.genfromtxt(self.maze_path, delimiter=',', dtype=np.uint8)

    def is_boundary(self):
        raise NotImplementedError