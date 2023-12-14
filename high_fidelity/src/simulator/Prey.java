package simulator;

import java.util.HashSet;
import java.util.Set;
import java.util.stream.Collectors;

public class Prey extends Agent implements Food {

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
    public Point getMoveUnchecked(WorldView worldView, double timespan) {
        double move_dist = Math.random() * this.speed * timespan;
        double move_angle = Math.random() * 2 * Math.PI;

        double move_x = Math.cos(move_angle) * move_dist;
        double move_y = Math.sin(move_angle) * move_dist;

        Point movement = new Point(move_x,move_y);

        return this.getLocation().add(movement);
    }

    @Override
    public Set<Food> canEat(Set<Food> foods) {
        return foods.stream().filter(food -> food instanceof Berry).collect(Collectors.toSet());
    }
}
