package simulator;

import java.util.Set;
import java.util.stream.Collectors;

public class Predator extends Animal {
    public Predator(Point location) {
        super(location);
    }

    @Override
    public Set<Food> canEat(Set<Food> foods) {
        return foods.stream().filter(food -> food instanceof Prey).collect(Collectors.toSet());
    }
}
