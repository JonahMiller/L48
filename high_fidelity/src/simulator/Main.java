package simulator;

import UI.UIFrame;

import javax.swing.*;

public class Main {
    public static void main(String[] args) {
        World world = new World(10,10,10);
        UIFrame frame = new UIFrame(world);
        frame.setVisible(true);
        int cnt = 0;
        while(true) {
            try {
                Thread.sleep(10);
                world.advanceTimeBy(0.5);
                frame.repaint();
            } catch (Exception e) {
                System.out.println(e);
            }
            if(cnt%10 == 0) {
                System.out.println("Iteration: " + cnt);
                System.out.println("Preys: " + world.getPreyLocations().size());
                System.out.println("Predators: " + world.getPredatorLocations().size());
                System.out.println("Berries: " + world.getBerryLocations().size());
                System.out.println();
            }
            cnt++;

        }
    }
}
