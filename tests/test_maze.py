from env.maze import Maze
from env.tiles import GreenTile, TileFactory
from env.tiles import WhiteTile
from env.tiles import OrangeTile
import numpy as np

def test_basic_maze_loading():
    '''Test Basic template loaded successfully'''
    maze_path = None
    maze = Maze(maze_path)
    maze_template = maze.maze_template

    assert maze_template.shape == (6, 6)

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

def test_maze_building():
    '''Tests maze building for the Basic Maze'''
    maze = Maze(maze_path=None)

    assert isinstance(maze.maze[0, 0], GreenTile)

def test_bounds_checking():
    '''Tests the boundaries for the Basic maze'''
    maze = Maze(maze_path=None)
    loc_1 = [-1, 7]
    loc_2 = [2, 2]
    loc_3 = [-5, 0]
    loc_4 = [2, 7]
    loc_5 = [0, 6]
    loc_6 = [5, 5]

    assert maze.is_out_of_bounds(loc_1) == True
    assert maze.is_out_of_bounds(loc_2) == False
    assert maze.is_out_of_bounds(loc_3) == True
    assert maze.is_out_of_bounds(loc_4) == True
    assert maze.is_out_of_bounds(loc_5) == True
    assert maze.is_out_of_bounds(loc_6) == False

def test_action_checker():
    '''Test if action checker is working'''
    maze = Maze(None)
    location = [0, 0]
    action_1 = 'Down'
    action_2 = 'Up'
    action_3 = 'Right'

    assert maze.action_viable(action_1, location) == True
    assert maze.action_viable(action_2, location) == False
    assert maze.action_viable(action_3, location) == False
    
