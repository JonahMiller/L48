package simulator;

import java.util.Set;
import java.util.stream.Collectors;

public class Prey extends Animal implements Food {

    private final double sustenanceValue;

    @Override
    public boolean exists() {
        return alive;
    }

    @Override
    public void consumed() {
        super.die();
    }

    public Prey(Point location,
                double startingEnergy,
                double starvationCoefficient,
                double stepEnergy,
                double eatingRadius,
                double reproductionEnergyThreshold,
                double speed,
                double sustenanceValue) {
        super(location,
              startingEnergy,
              starvationCoefficient,
              stepEnergy,
              eatingRadius,
              reproductionEnergyThreshold,
              speed);
        this.sustenanceValue = sustenanceValue;
    }

    @Override
    // TODO: Do we want to return the prey's energy instead?
    public double getSustenanceValue() {
        return sustenanceValue;
    }

    @Override
    public boolean canEat(Food food) {
        return food instanceof Berry;
    }
}
