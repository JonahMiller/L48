package simulator;

import java.util.HashSet;
import java.util.Set;

public abstract class Animal implements Cloneable {
    private Point location;

    protected boolean alive = true;
    protected final double viewRadius = 15; // WARNING: unused
    //    protected final double size = 10;
    protected double energy;
    protected final double starvationCoefficient;
    protected final double stepEnergy;
    protected final double eatingRadius;
    protected final double reproductionEnergyThreshold;
    protected final double speed;

    public Point getLocation() {
        return location;
    }

    public void starve(double timespan) {
        energy -= timespan * starvationCoefficient;
        if(energy <= 0) {
            die();
        }
    }

    public boolean canReproduce() {
        return (energy >= reproductionEnergyThreshold);
    }

    public Animal reproduce() {
        energy /= 2;
        try {
            Animal offspring = (Animal) this.clone();
            return offspring;
        } catch(CloneNotSupportedException e) {
            System.out.println(e);
            return null;
        }
    }

    public boolean isAlive() {
        return alive;
    }

    public void die() {
        this.alive = false;
    }

    public double getViewRadius() {
        return viewRadius;
    }

    public Animal(Point location,
                  double startingEnergy,
                  double starvationCoefficient,
                  double stepEnergy,
                  double eatingRadius,
                  double reproductionEnergyThreshold,
                  double speed) {
        this.location = location;
        this.energy = startingEnergy;
        this.starvationCoefficient = starvationCoefficient;
        this.stepEnergy = stepEnergy;
        this.eatingRadius = eatingRadius;
        this.reproductionEnergyThreshold = reproductionEnergyThreshold;
        this.speed = speed;
    }

    private boolean isWithinWorldBoundaries(Point p) {
        return World.minX <= p.x && p.x <= World.maxX && World.minY <= p.y && p.y <= World.maxY;
    }

    private boolean checkMove(Point move, double timespan) {
        return location.getDistance(move) <= speed * timespan && isWithinWorldBoundaries(move);
    }

    // This can be overridden in the subclasses later if we want differing behaviours
    public Point getMoveUnchecked(WorldView worldView, double timespan) {

        double move_dist = World.rng.nextDouble() * this.speed * timespan;
        double move_angle = World.rng.nextDouble() * 2 * Math.PI;

        double move_x = Math.cos(move_angle) * move_dist;
        double move_y = Math.sin(move_angle) * move_dist;

        Point movement = new Point(move_x, move_y);

        return this.getLocation().add(movement);
    }

    public final void move(WorldView worldView, double timespan) {
        Point newLocation = getMoveUnchecked(worldView, timespan);
        while(!checkMove(newLocation, timespan)) {
            newLocation = getMoveUnchecked(worldView, timespan);
        }
        energy -= location.getDistance(newLocation) * stepEnergy;
        location = newLocation;
    }

    public abstract boolean canEat(Food food);

    public Set<Food> eat(WorldView worldView) {
        Set<Food> meal = new HashSet<>();

        for(Food food : worldView.foods) {
            if(getLocation().getDistance(food.getLocation()) < eatingRadius && canEat(food)) {
                energy += food.getSustenanceValue();
                meal.add(food);
            }
        }
        return meal;
    }

    public double getEnergy() {
        return energy;
    }
}
