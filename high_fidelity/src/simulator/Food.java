package simulator;

public interface Food {
    public boolean exists();

    public void consumed();

    public Point getLocation();

    public double getSustenanceValue();

}