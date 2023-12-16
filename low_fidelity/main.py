import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation as animation
import random

from low_fidelity.animal import Predator, Prey, Food
from low_fidelity.animate import Animate
from low_fidelity.state import State

PREY_DEATH_FROM_PRED = 0.1

PREY_ENERGY = 20
PRED_ENERGY = 50

PREY_STEP_ENERGY = 2
PRED_STEP_ENERGY = 3

PREY_ENERGY_FROM_FOOD = 3
PRED_ENERGY_FROM_PREY = 10

PREY_REPRODUCTION_THRESHOLD = 15
PRED_REPRODUCTION_THRESHOLD = 40

PREY_REPRODUCTION_CHANCE = 0.3
PRED_REPRODUCTION_CHANCE = 0.1

NUM_FOOD = 1000


def empty_board(grid_x: int, grid_y: int):
    return State(grid_x, grid_y)


def populate_board(board: State, num_prey: int, num_pred: int):
    preys_x = np.random.randint(0, board.grid_x, size=num_prey)
    preys_y = np.random.randint(0, board.grid_y, size=num_prey)
    for x, y in zip(preys_x, preys_y):
        board.add_animal(Prey(x_0=x, y_0=y, 
                              energy=PREY_ENERGY, 
                              step_energy=PREY_STEP_ENERGY))

    preds_x = np.random.randint(0, board.grid_x, size=num_pred)
    preds_y = np.random.randint(0, board.grid_y, size=num_pred)
    for x, y in zip(preds_x, preds_y):
        board.add_animal(Predator(x_0=x, y_0=y, 
                                  energy=PRED_ENERGY, 
                                  step_energy=PRED_STEP_ENERGY))


def repopulate_board(board: State):
    preys_can_reproduce = 0
    preds_can_reproduce = 0
    for animal in board.view_animals():
        if isinstance(animal, Prey):
            if animal.energy >= PREY_REPRODUCTION_THRESHOLD:
                preys_can_reproduce += 1
        elif isinstance(animal, Predator):
            if animal.energy >= PRED_REPRODUCTION_THRESHOLD:
                preds_can_reproduce += 1
    new_preys = int(PREY_REPRODUCTION_CHANCE*preys_can_reproduce)
    new_preds = int(PRED_REPRODUCTION_CHANCE*preds_can_reproduce)
    populate_board(board, new_preys, new_preds)
    spawn_food(board)
    

def spawn_food(board: State):
    num_food = NUM_FOOD
    food_x = np.random.randint(0, board.grid_x, size=num_food)
    food_y = np.random.randint(0, board.grid_x, size=num_food)
    for x, y in zip(food_x, food_y):
        board.add_food(Food(x_0=x, y_0=y))


def animal_step(board: State):
    for animal in board.view_animals():
        board.step_animal(animal)


def interact(board: State):
    for coord in board.view_coords_with_items():
        preys = board.view_preys_by_loc(coord)
        preds = board.view_preds_by_loc(coord)
        foods = board.view_foods_by_loc(coord)
        n_preys, n_preds, n_foods = len(preys), len(preds), len(foods)

        # Prey eat the food available and get energy
        if n_foods > 0 and n_preys > 0:
            preys_to_eat = random.sample(preys, min(n_foods, n_preys))
            for prey in preys_to_eat:
                prey.energy += PREY_ENERGY_FROM_FOOD

        # Food is removed for the next turn
        for food in foods:
            board.remove_food(food)

        # Each prey dies wp. prey_death_from_pred for each pred,
        # they also die if they run out of energy
        n_eaten = 0
        for prey in preys:
            if np.random.rand() > (1 - PREY_DEATH_FROM_PRED) ** n_preds:
                board.remove_animal(prey)
                n_eaten += 1
            elif prey.energy <= 0:
                board.remove_animal(prey)

        # Preds gain energy from the prey eaten
        if n_eaten > 0 and n_preds > 0:
            preds_to_eat = random.sample(preds, min(n_eaten, n_preds))
            for pred in preds_to_eat:
                pred.energy += PRED_ENERGY_FROM_PREY

        # Preds die if they run out of energy
        for pred in preds:
            if pred.energy <= 0:
                board.remove_animal(pred)


def simulate(
    steps: int, grid_x: int, grid_y: int, init_prey: int, init_pred: int, save_all: bool = True
) -> list[State]:
    states = []

    board = empty_board(grid_x, grid_y)
    populate_board(board, init_prey, init_pred)
    spawn_food(board)

    for _ in range(steps):
        if len(board._preys) == 0 or len(board._preds) == 0:
            break
        if save_all:
            states.append(board.clone())

        animal_step(board)
        interact(board)
        repopulate_board(board)

    states.append(board.clone())

    return states


def main(steps, grid_x, grid_y, init_prey, init_pred):
    states = simulate(steps, grid_x, grid_y, init_prey, init_pred)

    num_preys, num_preds, num_foods = [], [], []
    preys_pos, preds_pos, foods_pos = [], [], []

    for board in states:
        num_preys.append(len(board._preys))
        num_preds.append(len(board._preds))
        num_foods.append(len(board._foods))
        preys_pos.append({coord: len(board.view_preys_by_loc(coord)) for coord in board.view_coords_with_prey()})
        preds_pos.append({coord: len(board.view_preds_by_loc(coord)) for coord in board.view_coords_with_pred()})
        foods_pos.append({coord: len(board.view_foods_by_loc(coord)) for coord in board.view_coords_with_food()})

    plt.plot(np.arange(len(num_preys)), num_preys, label="Preys")
    plt.plot(np.arange(len(num_preds)), num_preds, label="Predators")
    plt.ylim(0, None)
    plt.legend()
    plt.savefig("preys_preds.png", dpi=300)

    a = Animate(len(preds_pos), grid_x, grid_y, preys_pos, preds_pos, foods_pos)
    a.show()


if __name__ == "__main__":
    main(steps=500, grid_x=30, grid_y=30, init_prey=2000, init_pred=200)
