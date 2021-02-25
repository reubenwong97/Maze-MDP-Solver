from env.maze import Maze

def test_basic_maze_loading():
    maze_path = None
    test_maze = Maze(maze_path)

    assert test_maze.maze.shape == (6,6)