from env.maze import Maze
from env.tiles import TileFactory
from env.tiles import WhiteTile
from env.tiles import OrangeTile
import numpy as np

def test_basic_maze_loading():
    maze_path = None
    test_maze = Maze(maze_path)

    assert test_maze.maze.shape == (6,6)

def test_tile_creation():
    test_map = np.array([[0, 1, 2, 3], [3, 2, 1, 0]])
    # specify dtype=object or cannot store tiles
    tile_map = np.zeros(test_map.shape, dtype=object)
    factory = TileFactory()
    for i in range(test_map.shape[0]):
        for j in range(test_map.shape[1]):
            tile_map[i, j] = factory.create_tile(map_value=test_map[i, j], location=(i, j))

    assert isinstance(tile_map[0, 0], WhiteTile)
    assert isinstance(tile_map[1, 0], OrangeTile)