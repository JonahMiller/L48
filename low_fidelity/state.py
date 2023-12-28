import copy
from itertools import chain
from typing import Iterable, NamedTuple

from low_fidelity.animal import Animal, Food, Predator, Prey


class State:
    """Models the state of the simulation."""

    def __init__(self, grid_x: int, grid_y: int):
        self.grid_x = grid_x
        self.grid_y = grid_y

        # Internal representation of the state
        self._preys = {}
        self._preds = {}
        self._prey_grid = LazyAnimalGrid()
        self._pred_grid = LazyAnimalGrid()

        self._foods = {}
        self._food_grid = LazyFoodGrid()

    def clone(self):
        """Save a copy of the state."""
        return copy.deepcopy(self)

    # --- Provide different views of the state for efficient access ---
    def view_animals(self) -> Iterable[Animal]:
        """Returns a view of all animals in the state."""
        return chain(self._preys.values(), self._preds.values())

    def view_preys_by_loc(self, coord: tuple[int, int]):
        return list(self._prey_grid.get(coord, {}).values())

    def view_preds_by_loc(self, coord: tuple[int, int]):
        return list(self._pred_grid.get(coord, {}).values())

    def view_foods_by_loc(self, coord: tuple[int, int]):
        return list(self._food_grid.get(coord, {}).values())

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

    def view_coords_with_food(self):
        for coord in self._food_grid.keys():
            yield coord

    def view_food_count(self):
        return len(self._foods)

    def view_coords_with_items(self):
        for coord in self._prey_grid.keys() | self._pred_grid.keys() | self._food_grid.keys():
            yield coord

    def view_state_summary(self):
        num_preys = len(self._preys)
        num_preds = len(self._preds)
        num_foods = len(self._foods)
        preys_pos = {coord: len(self.view_preys_by_loc(coord)) for coord in self.view_coords_with_prey()}
        preds_pos = {coord: len(self.view_preds_by_loc(coord)) for coord in self.view_coords_with_pred()}
        foods_pos = {coord: len(self.view_foods_by_loc(coord)) for coord in self.view_coords_with_food()}
        return StateSummary(num_preys, num_preds, num_foods, preys_pos, preds_pos, foods_pos)

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

    def add_food(self, food: Food):
        if isinstance(food, Food):
            self._foods[id(food)] = food
            self._food_grid.add_food(food)
        else:
            raise ValueError(f"Unknown food type {type(food)}")

    def remove_food(self, food: Food):
        if isinstance(food, Food):
            del self._foods[id(food)]
            self._food_grid.remove_food(food)
        else:
            raise ValueError(f"Unknown food type {type(food)}")


class LazyAnimalGrid(dict[tuple[int, int], dict[int, Animal]]):
    """A lazy grid for storing animals.

    Only stores locations with at least one animal.
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


class LazyFoodGrid(dict[tuple[int, int], dict[int, Food]]):
    """A lazy grid for storing foods.

    Only stores locations with at least one food.
    """

    def __getitem__(self, coord):
        return self.setdefault(coord, {})

    def add_food(self, food: Food):
        """Adds a food item to the grid"""
        self[food.pos][id(food)] = food

    def remove_food(self, food: Food):
        """Remove a food item from the grid"""
        del self[food.pos][id(food)]
        if len(self[food.pos]) == 0:
            del self[food.pos]


class StateSummary(NamedTuple):
    """Summary of the state of the simulation."""

    num_preys: int
    num_preds: int
    num_foods: int
    preys_pos: dict[tuple[int, int], int]
    preds_pos: dict[tuple[int, int], int]
    foods_pos: dict[tuple[int, int], int]
