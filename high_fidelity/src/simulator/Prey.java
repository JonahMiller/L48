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
    // TODO: Do we want to return the prey's foodLevel instead?
    public double getSustenanceValue() {
        return sustenanceValue;
    }

    @Override
    public boolean canEat(Food food) {
        return food instanceof Berry;
    }
}
