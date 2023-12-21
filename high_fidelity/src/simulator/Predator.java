package simulator;

public class Predator extends Animal {
    public Predator(Point location,
                    double startingEnergy,
                    double starvationCoefficient,
                    double stepEnergy,
                    double eatingRadius,
                    double reproductionEnergyThreshold,
                    double speed) {
        super(location,
              startingEnergy,
              starvationCoefficient,
              stepEnergy,
              eatingRadius,
              reproductionEnergyThreshold,
              speed);
    }

    @Override
    public boolean canEat(Food food) {
        return food instanceof Prey;
    }
}
