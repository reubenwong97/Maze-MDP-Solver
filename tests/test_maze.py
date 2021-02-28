from env.maze import Maze
from env.tiles import GreenTile, TileFactory
from env.tiles import WhiteTile
from env.tiles import OrangeTile
from learner.value_iteration import ValueIterationLearner
import numpy as np

def test_basic_maze_loading():
    '''Test Basic template loaded successfully'''
    maze_path = 'default'
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
    maze = Maze(maze_path='default')

    assert isinstance(maze.maze[0, 0], GreenTile)

def test_bounds_checking():
    '''Tests the boundaries for the Basic maze'''
    maze = Maze(maze_path='default')
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

def test_state_provider():
    '''Test if correct state is returned is working'''
    maze = Maze('default')
    location = (0, 0)
    action_1 = 'Down'
    action_2 = 'Up'
    action_3 = 'Right'

    assert maze.get_new_state(action_1, location) == (1, 0)
    assert maze.get_new_state(action_2, location) == (0, 0)
    assert maze.get_new_state(action_3, location) == (0, 0)

def test_learnability():
    maze = Maze('default')
    location1 = (0, 0)
    location2 = (0, 1)

    assert maze.is_learnable_state(location1) == True
    assert maze.is_learnable_state(location2) == False 

def test_transitions():
    maze = Maze('default')
    location = (0, 0)
    action_1 = 'Down'
    action_2 = 'Up'

    transition_1 = np.array([[(1, 0), 0.8],
                             [(0, 0), 0.1],
                             [(0, 0), 0.1]], dtype=object)

    transition_2 = np.array([[(0, 0), 0.8],
                             [(0, 0), 0.1],
                             [(0, 0), 0.1]], dtype=object)

    assert np.all(maze.transitions(location, action_1) == transition_1)
    assert np.all(maze.transitions(location, action_2) == transition_2)

def test_location():
    env = Maze('default')
    learner = ValueIterationLearner(env)

    assert learner._to_location(6) == (1, 0)
    assert learner._to_location(9) == (1, 3)