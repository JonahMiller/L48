# loosely inspired by https://codereview.stackexchange.com/questions/230311/predator-prey-simulation

import numpy as np
import matplotlib.pyplot as plt

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
STAY = 4

prey_newborn = 0.4
pred_newborn = 0.1


class Animal(object):
    def __init__(self, x_0, y_0, species):
        self.life = 100
        self.x = x_0
        self.y = y_0
        self.species = species
        self.alive = True

    def die(self):
        self.alive = False

    def move(self, direction, grid_x, grid_y):
        if direction == LEFT:
            self.x += 1 if self.x > 0 else -1
        if direction == RIGHT:
            self.x -= 1 if self.x < grid_x -1 else -1
        if direction == UP:
            self.y += 1 if self.y < grid_y -1 else -1
        if direction == DOWN:
            self.y -= 1 if self.y > 0 else -1
        return self.x, self.y
            

def draw_grid(x_length, y_length):
    x_coords = np.arange(x_length)
    y_coords = np.arange(y_length)
    coords = np.transpose([np.tile(x_coords, len(y_coords)), np.repeat(y_coords, len(x_coords))])
    return coords


def populate_grid(grid_coords, num_prey, num_pred):
    rand_coords = np.random.permutation(grid_coords)
    prey_coords = rand_coords[:num_prey]
    pred_coords = rand_coords[num_prey:(num_prey + num_pred)]

    preys, preds = [], []

    for (x, y) in prey_coords:
        preys.append(Animal(x_0=x, y_0=y, species="prey"))
    for (x, y) in pred_coords:
        preds.append(Animal(x_0=x, y_0=y, species="pred"))

    return preys, preds

def animal_step(animal, grid_x, grid_y):
    animal_grid = {}
    directions = np.random.randint(0, 5, size=len(animal))
    for animal, direction in zip(animal, directions):
        pos = animal.move(direction, grid_x, grid_y)
        if pos not in animal_grid:
            animal_grid[pos] = 1
        else:
            animal_grid[pos] += 1
    return animal_grid

def interact(preys, preds, prey_grid, pred_grid):
    # If no preds, 2 prey reproduce wp. prey_newborn
    for coords, prey_here in prey_grid.items():
        if prey_here >= 2:
            if coords not in pred_grid:
                births = np.random.binomial(np.floor(prey_here / 2), prey_newborn)
                for _ in range(births):
                    preys.append(Animal(x_0=coords[0], y_0=coords[1], species="prey"))

    # If preds and prey at the same location, each prey dies wp. pred/(prey + pred)
    # If multiple preds, 2 preds reproduce wp. pred_newborn
    for coords, pred_here in pred_grid.items():
        if coords in prey_grid:
            # deaths = np.random.binomial(prey_grid[coords], pred_here/(prey_grid[coords] + pred_here))
            deaths = prey_grid[coords]
            
            # Kill off the prey
            for _ in range(deaths):
                for prey in preys:
                    if prey.x == coords[0] and prey.y == coords[1] and prey.alive:
                        prey.die()

        if pred_here >= 2:
            births = np.random.binomial(np.floor(pred_here / 2), pred_newborn)
            for _ in range(births):
                preds.append(Animal(x_0=coords[0], y_0=coords[1], species="pred"))

    return preys, preds

def remove_dead(preys, preds):
    return [prey for prey in preys if prey.alive], [pred for pred in preds if pred.alive]

def main(steps, grid_x, grid_y, init_prey, init_pred):
    coords = draw_grid(grid_x, grid_y)
    preys, preds = populate_grid(coords, init_prey, init_pred)
    num_preys, num_preds = [init_prey], [init_pred]
    for _ in range(steps):
        prey_grid = animal_step(preys, grid_x, grid_y)
        pred_grid = animal_step(preds, grid_x, grid_y)
        preys, preds = interact(preys, preds, prey_grid, pred_grid)
        preys, preds = remove_dead(preys, preds)
        num_preys.append(len(preys))
        num_preds.append(len(preds))

        if _ % 10 == 0:
            print(_)

    plt.plot(np.arange(steps+1), num_preys, label="prey")
    plt.plot(np.arange(steps+1), num_preds, label="pred")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main(steps=200, grid_x=30, grid_y=30, init_prey=20, init_pred=15)