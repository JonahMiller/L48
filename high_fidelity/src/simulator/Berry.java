package simulator;

public class Berry implements Food {

    private boolean alive = true;
    private final Point location;
    private static final double sustenanceValue = 50;

    @Override
    public boolean notEaten()
    {
        return alive;
    }


    @Override
    public void consumed() {
        alive = false;
    }

    public Berry(Point location) {
        this.location = location;
    }

    @Override
    public Point getLocation() {
        return location;
    }

    @Override
    public double getSustenanceValue() {
        return sustenanceValue;
    }
}
