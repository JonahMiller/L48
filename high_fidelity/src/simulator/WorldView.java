package simulator;

import java.util.Set;

public class WorldView {

    Set<Agent> animals;
    Set<Food> foods;

    WorldView (Set<Agent> animals, Set<Food> foods)
    {
        this.animals = animals;
        this.foods = foods;
    }
}
