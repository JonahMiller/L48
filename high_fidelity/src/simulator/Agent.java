package simulator;

import java.util.HashSet;
import java.util.Set;

public abstract class Agent implements Cloneable {
    private Point location;
    protected double speed = 100;

    protected boolean alive = true;
    protected final double viewRadius = 100;
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

    public Agent reproduce()  {
        if(foodLevel >= reproductionFoodLevel) {
            foodLevel /= 2;
            try {
                Agent offspring = (Agent) this.clone();
                return offspring;
            } catch (CloneNotSupportedException e) {
                System.out.println(e);
                return null;
            }
        } else {
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
    public Agent(Point location) {
        this.location = location;
    }

    private boolean isWithinWorldBoundaries(Point p) {
        return World.minX <= p.x && p.x <= World.maxX && World.minY <= p.y && p.y <= World.maxY;
    }

    private boolean checkMove(Point move, double timespan)
    {
        return location.getDistance(move) <= speed * timespan && isWithinWorldBoundaries(move);
    }

    public abstract Point getMoveUnchecked(WorldView worldView, double timespan);
    public final void move(WorldView worldView, double timespan) {
        Point newLocation = getMoveUnchecked(worldView, timespan);
        while (!checkMove(newLocation, timespan)){
            newLocation = getMoveUnchecked(worldView, timespan);
        };

        location = newLocation;

    }

    public abstract Set<Food> canEat(Set<Food> foods);
    public Set<Food> eat(WorldView worldView) {
        Set<Food> meal = new HashSet<>();

        Set<Food> eatable = canEat(worldView.foods);
        for (Food food : eatable)
        {
            if (getLocation().getDistance(food.getLocation()) < eatRadius)
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
