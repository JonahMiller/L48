package UI;

import javax.swing.*;
import java.awt.*;
import java.util.List;
import simulator.World;
import simulator.Point;

class Panel extends JPanel {

    private static final Dimension PANEL_SIZE = new Dimension(400, 400);

    private World world;

    public Panel(World world) {
        super();
        this.world = world;
    }

    @Override
    public Dimension getPreferredSize() {
        return PANEL_SIZE;
    }

    // We assume both the world and the screen is a rectangle
    public int transformCoordinate(double c) {
        return (int) Math.round((c - World.minX) / (World.maxX - World.minX) * getWidth());
    }

    @Override
    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        g.setColor(Color.WHITE);
        g.fillRect(0,0, getWidth(), getHeight());

        List<Point> preys = this.world.getPreyLocations();
        List<Point> predators = this.world.getPredatorLocations();
        List<Point> berries = this.world.getBerryLocations();

        int R = 8;

        // preys are green
        g.setColor(Color.GREEN);
        for(Point prey : preys) {
            g.fillOval(transformCoordinate(prey.x), transformCoordinate(prey.y), R, R);
        }

        // predators are red
        g.setColor(Color.RED);
        for(Point predator : predators) {
            g.fillOval(transformCoordinate(predator.x), transformCoordinate(predator.y), R, R);
        }

        // berries are blue
        g.setColor(Color.BLUE);
        for(Point berry : berries) {
            g.fillOval(transformCoordinate(berry.x), transformCoordinate(berry.y), R, R);
        }
    }
}
public class UIFrame extends JFrame {

    // constructor
    public UIFrame(World world) {
        super("Predator-prey");
        setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        getContentPane().add(new Panel(world), BorderLayout.CENTER);
        pack();
    }
}