package simulator;

import java.util.Set;

public class WorldView {

    Set<Animal> animals;
    Set<Food> foods;

    WorldView (Set<Animal> animals, Set<Food> foods)
    {
        this.animals = animals;
        this.foods = foods;
    }
}
