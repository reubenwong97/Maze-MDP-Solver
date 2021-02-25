COLORS = {
    "white": (255, 255, 255),
    "green": (0, 255, 0),
    "orange": (255, 204, 0),
    "grey": (153, 153, 153)
}

class Tile(object):
    def __init__(self, map_value:int, location: tuple):
        self.map_value = map_value
        self.location = location
        self._set_attributes()

    def _set_attributes(self):
        raise NotImplementedError

class WhiteTile(Tile):
    def __init__(self, map_value: int, location: tuple):
        super(WhiteTile, self).__init__(map_value, location)

    def _set_attributes(self):
        self.color = COLORS["white"]
        self.passable = True
        self.reward = 0

class GreyTile(Tile):
    def __init__(self, map_value: int, location: tuple):
        super(GreyTile, self).__init__(map_value, location)

    def _set_attributes(self):
        self.color = COLORS["grey"]
        self.passable = False
        self.reward = 0

class GreenTile(Tile):
    def __init__(self, map_value: int, location: tuple):
        super(GreenTile, self).__init__(map_value, location)

    def _set_attributes(self):
        self.color = COLORS["green"]
        self.passable = True
        self.reward = 1

class OrangeTile(Tile):
    def __init__(self, map_value: int, location: tuple):
        super(OrangeTile, self).__init__(map_value, location)

    def _set_attributes(self):
        self.color = COLORS["orange"]
        self.passable = True
        self.reward = -1

class TileFactory(object):
    def create_tile(self, map_value: int, location: tuple) -> Tile:
        if map_value == 0:
            return WhiteTile(map_value, location)
        elif map_value == 1:
            return GreyTile(map_value, location)
        elif map_value == 2:
            return GreenTile(map_value, location)
        elif map_value == 3:
            return OrangeTile(map_value, location)
        else:
            raise ValueError("Values in map restricted to 0, 1, 2 or 3 only")