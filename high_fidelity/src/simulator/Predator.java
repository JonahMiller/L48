package simulator;

import java.util.Set;
import java.util.stream.Collectors;

public class Predator extends Animal {
    public Predator(Point location) {
        super(location);
    }

    @Override
    public boolean canEat(Food food) {
        return food instanceof Prey;
    }
}
