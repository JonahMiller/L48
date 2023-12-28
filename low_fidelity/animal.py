import numpy as np

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
STAY = 4


class Animal:
    def __init__(self, x_0: int, y_0: int, energy: float, step_energy: float):
        self.x = x_0
        self.y = y_0
        self.energy = energy
        self.step_energy = step_energy

    @property
    def pos(self):
        return self.x, self.y

    def step(self, grid_x: int, grid_y: int):
        direction = np.random.randint(0, 5)
        self.energy -= self.step_energy
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


class Food:
    def __init__(self, x_0: int, y_0: int):
        self.x = x_0
        self.y = y_0

    @property
    def pos(self):
        return self.x, self.y
