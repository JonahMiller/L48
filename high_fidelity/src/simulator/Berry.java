package simulator;

public class Berry implements Food {

    private boolean alive = true;
    private final Point location;
    private final double sustenanceValue;

    @Override
    public boolean exists() {
        return alive;
    }


    @Override
    public void consumed() {
        alive = false;
    }

    public Berry(Point location, double sustenanceValue) {
        this.location = location;
        this.sustenanceValue = sustenanceValue;
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
