package simulator;

import java.util.HashSet;
import java.util.Set;

public class Prey extends Agent implements Food {

    private static final double sustenanceValue = 50;

    @Override
    public boolean notEaten() {
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
    public Point getMoveUnchecked(WorldView worldView, double timespan) {
        double move_dist = Math.random() * this.speed * timespan;
        double move_angle = Math.random() * 2 * Math.PI;

        double move_x = Math.cos(move_angle) * move_dist;
        double move_y = Math.sin(move_angle) * move_dist;

        Point movement = new Point(move_x,move_y);

        return this.getLocation().add(movement);
    }

    @Override
    public Set<Food> eat(WorldView worldView) {
        Set<Food> meal = new HashSet<>();

        for (Food food : worldView.foods)
        {
            if (food instanceof Berry && getLocation().getDistance(food.getLocation()) < eatRadius)
            {
                foodLevel += food.getSustenanceValue();
                meal.add(food);
            }
        }
        return meal;
    }
}
