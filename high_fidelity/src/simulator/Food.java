package simulator;

public interface Food {
    public boolean notEaten();

    public void consumed();
    public Point getLocation();
    public double getSustenanceValue();

}
