from itertools import chain
from typing import Iterable

from low_fidelity.animal import Animal, Predator, Prey


class State:
    """Models the state of the simulation."""

    def __init__(self, grid_x: int, grid_y: int):
        self.grid_x = grid_x
        self.grid_y = grid_y

        # Internal representation of the state
        self._preys = {}
        self._preds = {}
        self._prey_grid = LazyGrid()
        self._pred_grid = LazyGrid()

    # --- Provide different views of the state for efficient access ---
    def view_animals(self) -> Iterable[Animal]:
        """Returns a view of all animals in the state."""
        return chain(self._preys.values(), self._preds.values())

    def view_preys_by_loc(self, coord: tuple[int, int]):
        return list(self._prey_grid.get(coord, {}).values())

    def view_preds_by_loc(self, coord: tuple[int, int]):
        return list(self._pred_grid.get(coord, {}).values())

    def view_coords(self):
        for x in range(self.grid_x):
            for y in range(self.grid_y):
                yield x, y

    def view_coords_with_prey(self):
        for coord in self._prey_grid.keys():
            yield coord

    def view_coords_with_pred(self):
        for coord in self._pred_grid.keys():
            yield coord

    def view_coords_with_animal(self):
        for coord in self._prey_grid.keys() | self._pred_grid.keys():
            yield coord

    # --- Methods for modifying the state ---
    def add_animal(self, animal: Animal):
        if isinstance(animal, Prey):
            self._preys[id(animal)] = animal
            self._prey_grid.add_animal(animal)
        elif isinstance(animal, Predator):
            self._preds[id(animal)] = animal
            self._pred_grid.add_animal(animal)
        else:
            raise ValueError(f"Unknown animal type {type(animal)}")

    def remove_animal(self, animal: Animal):
        if isinstance(animal, Prey):
            del self._preys[id(animal)]
            self._prey_grid.remove_animal(animal)
        elif isinstance(animal, Predator):
            del self._preds[id(animal)]
            self._pred_grid.remove_animal(animal)
        else:
            raise ValueError(f"Unknown animal type {type(animal)}")

    def step_animal(self, animal: Animal):
        if isinstance(animal, Prey):
            self._prey_grid.remove_animal(animal)
            animal.step(self.grid_x, self.grid_y)
            self._prey_grid.add_animal(animal)
        elif isinstance(animal, Predator):
            self._pred_grid.remove_animal(animal)
            animal.step(self.grid_x, self.grid_y)
            self._pred_grid.add_animal(animal)
        else:
            raise ValueError(f"Unknown animal type {type(animal)}")


class LazyGrid(dict[tuple[int, int], dict[int, Animal]]):
    """A lazy grid for storing animals.

    Only stores locations with more than one animal.
    """

    def __getitem__(self, coord):
        return self.setdefault(coord, {})

    def add_animal(self, animal: Animal):
        """Adds an animal to the grid."""
        self[animal.pos][id(animal)] = animal

    def remove_animal(self, animal: Animal):
        """Removes an animal from the grid."""
        del self[animal.pos][id(animal)]
        if len(self[animal.pos]) == 0:
            del self[animal.pos]
