# loosely inspired by https://codereview.stackexchange.com/questions/230311/predator-prey-simulation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation as animation

from low_fidelity.animal import Predator, Prey
from low_fidelity.animate import Animate
from low_fidelity.state import State

PREY_BIRTH = 0.2
PRED_DEATH = 0.1
PREY_DEATH_FROM_PRED = 0.2
PRED_BIRTH_FROM_PREY = 0.5


def empty_board(grid_x: int, grid_y: int):
    return State(grid_x, grid_y)


def populate_board(board: State, num_prey: int, num_pred: int):
    preys_x = np.random.randint(0, board.grid_x, size=num_prey)
    preys_y = np.random.randint(0, board.grid_y, size=num_prey)
    for x, y in zip(preys_x, preys_y):
        board.add_animal(Prey(x_0=x, y_0=y))

    preds_x = np.random.randint(0, board.grid_x, size=num_pred)
    preds_y = np.random.randint(0, board.grid_y, size=num_pred)
    for x, y in zip(preds_x, preds_y):
        board.add_animal(Predator(x_0=x, y_0=y))


def animal_step(board: State):
    for animal in board.view_animals():
        board.step_animal(animal)


def interact(board: State):
    for coord in board.view_coords_with_animal():
        preys = board.view_preys_by_loc(coord)
        preds = board.view_preds_by_loc(coord)
        n_preys, n_preds = len(preys), len(preds)

        # 2 prey reproduce wp. prey_newborn
        births = np.random.binomial(n_preys // 2, PREY_BIRTH)
        for _ in range(births):
            board.add_animal(Prey(x_0=coord[0], y_0=coord[1]))

        # Each prey dies wp. prey_death_from_pred for each pred
        n_eaten = 0
        for prey in preys:
            if np.random.rand() > (1 - PREY_DEATH_FROM_PRED) ** n_preds:
                board.remove_animal(prey)
                n_eaten += 1

        # Preds reproduce proportional to n_eaten
        births = np.random.binomial(n_eaten, PRED_BIRTH_FROM_PREY)
        for _ in range(births):
            board.add_animal(Predator(x_0=coord[0], y_0=coord[1]))

        # Preds naturally die wp. pred_death
        for pred in preds:
            if np.random.rand() < PRED_DEATH:
                board.remove_animal(pred)


def main(steps, grid_x, grid_y, init_prey, init_pred):
    board = empty_board(grid_x, grid_y)
    populate_board(board, init_prey, init_pred)

    num_preys, num_preds = [], []
    preys_pos, preds_pos = [], []

    def snapshot(board: State):
        num_preys.append(len(board._preys))
        num_preds.append(len(board._preds))
        preys_pos.append({coord: len(board.view_preys_by_loc(coord)) for coord in board.view_coords_with_prey()})
        preds_pos.append({coord: len(board.view_preds_by_loc(coord)) for coord in board.view_coords_with_pred()})

    snapshot(board)
    for _ in range(steps):
        animal_step(board)
        interact(board)
        snapshot(board)

        # if len(board._preys) == 0 or len(board._preds) == 0:
        #     break

    plt.plot(np.arange(len(num_preys)), num_preys, label="Preys")
    plt.plot(np.arange(len(num_preds)), num_preds, label="Predators")
    plt.ylim(0, None)
    plt.legend()
    plt.savefig("preys_preds.png", dpi=300)

    a = Animate(len(preds_pos), grid_x, grid_y, preys_pos, preds_pos)
    a.show()


if __name__ == "__main__":
    main(steps=100, grid_x=30, grid_y=30, init_prey=2000, init_pred=200)
