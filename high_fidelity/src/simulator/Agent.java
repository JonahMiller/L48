package simulator;

import java.util.HashSet;
import java.util.Set;

public abstract class Agent {
    private Point location;
    protected double speed = 10;

    protected boolean alive = true;
    protected final double viewRadius = 100;
    protected final double size = 10;
    protected double foodLevel = 50;

    protected double eatRadius = 10;

    public Point getLocation() {
        return location;
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

    private boolean checkMove(Point move, double timespan)
    {
        return location.getDistance(move) <= speed * timespan;
    }

    public abstract Point getMoveUnchecked(WorldView worldView, double timespan);
    public final void move(WorldView worldView, double timespan) {
        Point newLocation = getMoveUnchecked(worldView, timespan);
        if (checkMove(newLocation, timespan))
            location = newLocation;
        else throw new UnsupportedOperationException("Invalid move");
    }

    public abstract Set<Food> eat(WorldView worldView);

    public double getFoodLevel() {
        return foodLevel;
    }
}
