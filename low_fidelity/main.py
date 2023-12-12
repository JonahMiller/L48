# loosely inspired by https://codereview.stackexchange.com/questions/230311/predator-prey-simulation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as animation
from IPython.display import HTML
plt.rcParams["animation.html"] = "jshtml"

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
            if self.x != 0:
                self.x -= 1
            else:
                self.x = grid_x - 1
        if direction == RIGHT:
            if self.x != grid_x - 1:
                self.x += 1
            else:
                self.x = 0
        if direction == DOWN:
            if self.y != 0:
                self.y -= 1
            else:
                self.y = grid_y - 1
        if direction == UP:
            if self.y != grid_y - 1:
                self.y += 1
            else:
                self.y = 0
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
    preys_grid, preds_grid = {}, {}

    for (x, y) in prey_coords:
        preys.append(Animal(x_0=x, y_0=y, species="prey"))
        preys_grid[(x, y)] = 1
    for (x, y) in pred_coords:
        preds.append(Animal(x_0=x, y_0=y, species="pred"))
        preds_grid[(x, y)] = 1

    return preys, preds, preys_grid, preds_grid

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

def interact(preys, preds, preys_grid, preds_grid):
    # If no preds, 2 prey reproduce wp. prey_newborn
    for coords, prey_here in preys_grid.items():
        if prey_here >= 2:
            if coords not in preds_grid:
                births = np.random.binomial(np.floor(prey_here / 2), prey_newborn)
                preys_grid[coords] += births
                for _ in range(births):
                    preys.append(Animal(x_0=coords[0], y_0=coords[1], species="prey"))

    # If preds and prey at the same location, each prey dies wp. pred/(prey + pred)
    # If multiple preds, 2 preds reproduce wp. pred_newborn
    for coords, pred_here in preds_grid.items():
        if coords in preys_grid:
            deaths = np.random.binomial(preys_grid[coords], pred_here/(preys_grid[coords] + pred_here))
            if preys_grid[coords] == deaths:
                del preys_grid[coords]
            else:
                preys_grid[coords] -= deaths
            
            # Kill off the prey
            for _ in range(deaths):
                for prey in preys:
                    if prey.x == coords[0] and prey.y == coords[1] and prey.alive:
                        prey.die()

        if pred_here >= 2:
            births = np.random.binomial(np.floor(pred_here / 2), pred_newborn)
            preds_grid[coords] += births
            for _ in range(births):
                preds.append(Animal(x_0=coords[0], y_0=coords[1], species="pred"))

    return preys, preds, preys_grid, preds_grid

def remove_dead(preys, preds):
    return [prey for prey in preys if prey.alive], [pred for pred in preds if pred.alive]

def main(steps, grid_x, grid_y, init_prey, init_pred):
    coords = draw_grid(grid_x, grid_y)
    preys, preds, preys_grid, preds_grid = populate_grid(coords, init_prey, init_pred)

    num_preys, num_preds = [init_prey], [init_pred]
    preys_pos, preds_pos = [preys_grid], [preds_grid]

    for _ in range(steps):
        preys_grid = animal_step(preys, grid_x, grid_y)
        preds_grid = animal_step(preds, grid_x, grid_y)
        preys, preds, preys_grid, preds_grid = interact(preys, preds, preys_grid, preds_grid)
        preys, preds = remove_dead(preys, preds)

        num_preys.append(len(preys))
        num_preds.append(len(preds))

        preys_pos.append(preys_grid)
        preds_pos.append(preds_grid)

    plt.plot(np.arange(steps+1), num_preys, label="Preys")
    plt.plot(np.arange(steps+1), num_preds, label="Predators")
    plt.legend()
    plt.show()

    a = Animate(steps, grid_x, grid_y, preys_pos, preds_pos)
    a.show()


class Animate:
    def __init__(self, steps, grid_x, grid_y, preys_pos, preds_pos):
        self.steps = steps
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.preys_pos = preys_pos
        self.preds_pos = preds_pos

        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, self.grid_y - 1)
        self.ax.set_ylim(0, self.grid_y - 1)
        self.ax.set_xticks(np.arange(0, self.grid_x - 1))
        self.ax.set_yticks(np.arange(0, self.grid_y - 1))
        plt.grid(True)

        self.prey_x, self.prey_y = self.coords_at_step(self.preys_pos[0])
        self.pred_x, self.pred_y = self.coords_at_step(self.preds_pos[0])
        self.prey_point, = self.ax.plot(self.prey_x, self.prey_y, 'bo', label="Preys")
        self.pred_point, = self.ax.plot(self.pred_x, self.pred_y, 'ro', label="Predators")
        self.fig.legend()
        self.txt = self.ax.text(0.1, 0.1,'', ha='center', va='center', alpha=0.8,
                    transform=self.ax.transAxes, fontdict={'color':'black', 'backgroundcolor': 'white', 'size': 10})

    def coords_at_step(self, grid):
        x = []
        y = []
        for i in grid.keys():
            x.append(i[0])
            y.append(i[1])
        return x, y

    def animate(self, i):
        self.prey_x, self.prey_y = self.coords_at_step(self.preys_pos[i])
        self.pred_x, self.pred_y = self.coords_at_step(self.preds_pos[i])
        self.prey_point.set_data(self.prey_x, self.prey_y)
        self.pred_point.set_data(self.pred_x, self.pred_y)
        self.txt.set_text(f"Step: {i}\nPreys: {sum(v for v in self.preys_pos[i].values())}\nPredators: {sum(v for v in self.preds_pos[i].values())}")
        return self.prey_point, self.pred_point, self.txt
    
    def show(self):
        anim = animation.FuncAnimation(fig=self.fig, func=self.animate, frames=self.steps,
                                       repeat=True, save_count=10, blit=True)
        HTML(anim.to_jshtml())
        anim.save('animation.gif', fps=10)


if __name__ == '__main__':
    main(steps=100, grid_x=20, grid_y=20, init_prey=40, init_pred=20)