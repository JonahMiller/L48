package simulator;

import java.util.Set;
import java.util.stream.Collectors;

public class Prey extends Animal implements Food {

    private static final double sustenanceValue = 50;

    @Override
    public boolean exists() {
        return alive;
    }

    @Override
    public void consumed() {
        super.die();
    }

    public Prey(Point location) {
        super(location);
    }

    @Override
    public double getSustenanceValue() {
        return sustenanceValue;
    }

    @Override
    public Set<Food> canEat(Set<Food> foods) {
        return foods.stream().filter(food -> food instanceof Berry).collect(Collectors.toSet());
    }
}
