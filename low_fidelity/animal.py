import numpy as np

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
STAY = 4

PREY_NEWBORN = 0.4
PRED_NEWBORN = 0.1


class Animal:
    def __init__(self, x_0: int, y_0: int):
        self.x = x_0
        self.y = y_0
        self.alive = True

    @property
    def pos(self):
        return self.x, self.y

    def die(self):
        self.alive = False

    def step(self, grid_x: int, grid_y: int):
        direction = np.random.randint(0, 5)
        if direction == LEFT:
            self.x = (self.x - 1) % grid_x
        if direction == RIGHT:
            self.x = (self.x + 1) % grid_x
        if direction == DOWN:
            self.y = (self.y - 1) % grid_y
        if direction == UP:
            self.y = (self.y + 1) % grid_y
        if direction == STAY:
            pass
        return self.pos


class Prey(Animal):
    ...


class Predator(Animal):
    ...
