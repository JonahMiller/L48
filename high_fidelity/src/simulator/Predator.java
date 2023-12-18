package simulator;

import java.util.Set;
import java.util.stream.Collectors;

public class Predator extends Animal {
    public Predator(Point location, double startingFoodLevel, double starvationCoefficient, double eatingRadius, double reproductionFoodLevel, double speed) {
        super(location, startingFoodLevel, starvationCoefficient, eatingRadius, reproductionFoodLevel, speed);
    }

    @Override
    public boolean canEat(Food food) {
        return food instanceof Prey;
    }
}
