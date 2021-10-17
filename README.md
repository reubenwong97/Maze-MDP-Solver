# Maze MDP Solver
## Maze Implementation
Mazes are implemented as numpy matrices of tiles. Tile objects hold all information regarding the nature of the tile, like whether it is passable, its colour and its reward.
## Custom Maze
Currently, there are 2 mazes provided here under `env/mazes/`, namely a default maze and a hard maze. The first step to implementing a custom maze is to generate a text file of your own and place it there in the directory. Next, you will need to implement the behaviour of any new tiles you have created. Do this in the `env/tiles.py` file. Be sure to update the `TileFactory`class so that it is able to correctly produce your tile.

Lastly, you will need to implement your own transition function in the `Maze` class.
## Solvers
Two solvers are provided here, and you can see how to use them in `main.py`.
## Technical Report
Please see the included technical report for a detailed introduction on the topic.
