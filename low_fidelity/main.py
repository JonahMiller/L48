import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation as animation

import sys
sys.path.append("..")
from low_fidelity.animal import Food, Predator, Prey
from low_fidelity.animate import Animate
from low_fidelity.state import State, StateSummary


@dataclass
class HyperParams:
    STEPS: int = 1000

    GRID_X: int = 20
    GRID_Y: int = 20

    INIT_PREY: int = 200
    INIT_PRED: int = 20

    # Number of food added randomly each step, and the maximal
    # amount of food allowed on the board
    NUM_FOOD: int = 10
    MAX_FOOD: int = 500

    # If a prey and n predators meet, the chance of prey being
    # eaten is 1 - (1 - PREY_DEATH_FROM_PRED)^n
    PREY_DEATH_FROM_PRED: float = 1

    # Starting energy for each animal. If their energy falls
    # <= 0 they die.
    PREY_ENERGY: float = 50
    PRED_ENERGY: float = 50

    # How much energy each animal expends per step
    PREY_STEP_ENERGY: float = 2
    PRED_STEP_ENERGY: float = 2

    # How much energy each animal gets from eating
    PREY_ENERGY_FROM_FOOD: float = 50
    PRED_ENERGY_FROM_PREY: float = 50

    # How much energy each animal needs to be able to reproduce
    PREY_REPRODUCTION_THRESHOLD: float = 100
    PRED_REPRODUCTION_THRESHOLD: float = 100

    # If they reach the energy threshold above, what proportion
    # of animals will reproduce
    PREY_REPRODUCTION_CHANCE: float = 1.0
    PRED_REPRODUCTION_CHANCE: float = 1.0

    # Animals spawn spontaneously even if there are no other animals
    # of their type to reproduce
    PREY_SPAWN_RATE: float = 0.05
    PRED_SPAWN_RATE: float = 0.05


def empty_board(hp: HyperParams):
    return State(hp.GRID_X, hp.GRID_Y)


def populate_board(board: State, hp: HyperParams, num_prey: int, num_pred: int):
    preys_x = np.random.randint(0, board.grid_x, size=num_prey)
    preys_y = np.random.randint(0, board.grid_y, size=num_prey)
    for x, y in zip(preys_x, preys_y):
        board.add_animal(Prey(x_0=x, y_0=y, energy=hp.PREY_ENERGY, step_energy=hp.PREY_STEP_ENERGY))

    preds_x = np.random.randint(0, board.grid_x, size=num_pred)
    preds_y = np.random.randint(0, board.grid_y, size=num_pred)
    for x, y in zip(preds_x, preds_y):
        board.add_animal(Predator(x_0=x, y_0=y, energy=hp.PRED_ENERGY, step_energy=hp.PRED_STEP_ENERGY))


def repopulate_board(board: State, hp: HyperParams):
    prey_offsprings = []
    pred_offsprings = []
    for animal in board.view_animals():
        if isinstance(animal, Prey):
            if animal.energy >= hp.PREY_REPRODUCTION_THRESHOLD:
                if np.random.rand() < hp.PREY_REPRODUCTION_CHANCE:
                    # The energy is split equally between the animal
                    # and the offspring
                    animal.energy /= 2
                    prey_offsprings.append(
                        Prey(x_0=animal.x, y_0=animal.y, energy=animal.energy, step_energy=hp.PREY_STEP_ENERGY)
                    )
        elif isinstance(animal, Predator):
            if animal.energy >= hp.PRED_REPRODUCTION_THRESHOLD:
                if np.random.rand() < hp.PRED_REPRODUCTION_CHANCE:
                    # The energy is split equally between the animal
                    # and the offspring
                    animal.energy /= 2
                    prey_offsprings.append(
                        Predator(x_0=animal.x, y_0=animal.y, energy=animal.energy, step_energy=hp.PRED_STEP_ENERGY)
                    )
    for prey in prey_offsprings:
        board.add_animal(prey)
    for pred in pred_offsprings:
        board.add_animal(pred)

    # Spontaneously spawn some animals so they don't die out
    spontaneous_prey_count = np.random.poisson(lam=hp.PREY_SPAWN_RATE)
    spontaneous_pred_count = np.random.poisson(lam=hp.PRED_SPAWN_RATE)
    populate_board(board, hp, num_prey=spontaneous_prey_count, num_pred=spontaneous_pred_count)

    spawn_food(board, hp)


def spawn_food(board: State, hp: HyperParams):
    num_food = min(hp.NUM_FOOD, hp.MAX_FOOD - board.view_food_count())
    food_x = np.random.randint(0, board.grid_x, size=num_food)
    food_y = np.random.randint(0, board.grid_x, size=num_food)
    for x, y in zip(food_x, food_y):
        board.add_food(Food(x_0=x, y_0=y))


def animal_step(board: State):
    for animal in board.view_animals():
        board.step_animal(animal)


def interact(board: State, hp: HyperParams):
    for coord in board.view_coords_with_items():
        preys = board.view_preys_by_loc(coord)
        preds = board.view_preds_by_loc(coord)
        foods = board.view_foods_by_loc(coord)
        n_preys, n_preds, n_foods = len(preys), len(preds), len(foods)

        # Prey eat the food available and get energy
        if n_foods > 0 and n_preys > 0:
            food_eaten = min(n_foods, n_preys)
            preys_to_eat = random.sample(preys, food_eaten)
            for prey in preys_to_eat:
                prey.energy += hp.PREY_ENERGY_FROM_FOOD

            for food in foods[0:food_eaten]:
                board.remove_food(food)

        # Food is removed for the next turn
        # for food in foods:
        #     board.remove_food(food)

        # Each prey dies wp. prey_death_from_pred for each pred,
        # they also die if they run out of energy
        n_eaten = 0
        for prey in preys:
            if np.random.rand() > (1 - hp.PREY_DEATH_FROM_PRED) ** n_preds:
                board.remove_animal(prey)
                n_eaten += 1
            elif prey.energy <= 0:
                board.remove_animal(prey)

        # Preds gain energy from the prey eaten
        if n_eaten > 0 and n_preds > 0:
            preds_to_eat = random.sample(preds, min(n_eaten, n_preds))
            for pred in preds_to_eat:
                pred.energy += hp.PRED_ENERGY_FROM_PREY

        # Preds die if they run out of energy
        for pred in preds:
            if pred.energy <= 0:
                board.remove_animal(pred)


def simulate(hp: HyperParams, save_all: bool = True) -> list[StateSummary]:
    states = []

    board = empty_board(hp)
    populate_board(board, hp, hp.INIT_PREY, hp.INIT_PRED)
    spawn_food(board, hp)

    for _ in range(hp.STEPS):
        if save_all:
            states.append(board.view_state_summary())

        animal_step(board)
        interact(board, hp)
        repopulate_board(board, hp)

    states.append(board.view_state_summary())

    return states


def main(hp: HyperParams):
    states = simulate(hp)

    num_preys, num_preds, num_foods = [], [], []
    preys_pos, preds_pos, foods_pos = [], [], []

    for summary in states:
        num_preys.append(summary.num_preys)
        num_preds.append(summary.num_preds)
        num_foods.append(summary.num_foods)
        preys_pos.append(summary.preys_pos)
        preds_pos.append(summary.preds_pos)
        foods_pos.append(summary.foods_pos)

    plt.plot(np.arange(len(num_preys)), num_preys, label="Preys")
    plt.plot(np.arange(len(num_preds)), num_preds, label="Predators")
    # plt.plot(np.arange(len(num_foods)), num_foods, label="Food")
    plt.grid()
    plt.xlabel("Steps")
    plt.ylabel("Population size")
    plt.ylim(0, None)
    plt.legend()
    plt.savefig("preys_preds.png", dpi=300)

    print("avg preys", np.mean(num_preys))
    print("avg preds", np.mean(num_preds))

    # a = Animate(len(preds_pos), hp.GRID_X, hp.GRID_Y, preys_pos, preds_pos, foods_pos)
    # a.show()


if __name__ == "__main__":
    # hp = HyperParams()  # Use default hyperparameters

    hp = HyperParams(
        STEPS = 1000,
        GRID_X = 10,
        GRID_Y = 10,
        INIT_PREY = 200,
        INIT_PRED = 20,
        NUM_FOOD = 250,
        MAX_FOOD = 1000,
        PREY_DEATH_FROM_PRED = 0.4,
        PREY_ENERGY = 20,
        PRED_ENERGY = 50,
        PREY_STEP_ENERGY = 2,
        PRED_STEP_ENERGY = 3,
        PREY_ENERGY_FROM_FOOD = 3,
        PRED_ENERGY_FROM_PREY = 10,
        PREY_REPRODUCTION_THRESHOLD = 15,
        PRED_REPRODUCTION_THRESHOLD = 20,
        PREY_REPRODUCTION_CHANCE = 0.3,
        PRED_REPRODUCTION_CHANCE = 0.1,
        PREY_SPAWN_RATE = 0,
        PRED_SPAWN_RATE = 0)

    main(hp)
