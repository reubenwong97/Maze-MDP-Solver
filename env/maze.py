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