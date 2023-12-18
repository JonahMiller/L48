package simulator;

import java.util.HashSet;
import java.util.Set;

public abstract class Animal implements Cloneable {
    private Point location;
    protected double speed = 100;

    protected boolean alive = true;
    protected final double viewRadius = 15;
    protected final double size = 10;
    protected double foodLevel = 50;
    protected final double reproductionFoodLevel = 50;

    protected final double eatRadius = 10;

    protected final double starvationCoefficient = 2;

    public Point getLocation() {
        return location;
    }

    public void starve(double timespan)
    {
        foodLevel -= timespan * starvationCoefficient;
        if (foodLevel<=0) die();
    }

    public boolean canReproduce() {
        return (foodLevel >= reproductionFoodLevel);
    }

    public Animal reproduce()  {
        foodLevel /= 2;
        try {
            Animal offspring = (Animal) this.clone();
            return offspring;
        } catch (CloneNotSupportedException e) {
            System.out.println(e);
            return null;
        }
    }

    public boolean isAlive()
    {
        return alive;
    }

    public void die()
    {
        this.alive = false;
    }
    public double getViewRadius() {return viewRadius;}
    public Animal(Point location) {
        this.location = location;
    }

    private boolean isWithinWorldBoundaries(Point p) {
        return World.minX <= p.x && p.x <= World.maxX && World.minY <= p.y && p.y <= World.maxY;
    }

    private boolean checkMove(Point move, double timespan)
    {
        return location.getDistance(move) <= speed * timespan && isWithinWorldBoundaries(move);
    }

    // This can be overridden in the subclasses later if we want differing behaviours
    public Point getMoveUnchecked(WorldView worldView, double timespan) {

        double move_dist = World.rng.nextDouble() * this.speed * timespan;
        double move_angle = World.rng.nextDouble() * 2 * Math.PI;

        double move_x = Math.cos(move_angle) * move_dist;
        double move_y = Math.sin(move_angle) * move_dist;

        Point movement = new Point(move_x,move_y);

        return this.getLocation().add(movement);
    }
    public final void move(WorldView worldView, double timespan) {
        Point newLocation = getMoveUnchecked(worldView, timespan);
        while (!checkMove(newLocation, timespan)){
            newLocation = getMoveUnchecked(worldView, timespan);
        };

        location = newLocation;

    }

    public abstract boolean canEat(Food food);
    public Set<Food> eat(WorldView worldView) {
        Set<Food> meal = new HashSet<>();

        for (Food food : worldView.foods)
        {
            if (getLocation().getDistance(food.getLocation()) < eatRadius && canEat(food))
            {
                foodLevel += food.getSustenanceValue();
                meal.add(food);
            }
        }
        return meal;
    }

    public double getFoodLevel() {
        return foodLevel;
    }
}
