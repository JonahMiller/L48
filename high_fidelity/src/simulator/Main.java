package simulator;

import UI.UIFrame;

import javax.swing.*;

public class Main {
    public static void main(String[] args) {
        World world = new World(ReproductionType.FOOD,
                                100,
                                20,
                                100,
                                500,
                                50,
                                50,
                                50,
                                50,
                                2,
                                2,
                                0,
                                0,
                                25,
                                1.0,
                                100,
                                100,
                                1,
                                1,
                                80,
                                0.05,
                                0.05,
                                10);
        UIFrame frame = new UIFrame(world);
        frame.setVisible(true);
        int cnt = 0;
        final int FPS = 100;
        while(true) {
            try {
                long start = System.nanoTime();
                world.advanceTimeBy(0.5);
                frame.repaint();
                long done = System.nanoTime();
                if(done - start < (1e9 / FPS)) {
                    long to_sleep = Math.round((1e9 / FPS - (done - start)) / 1e6);
                    Thread.sleep(to_sleep);
                }
            } catch(Exception e) {
                System.out.println(e);
            }
            int preys = world.getPreyLocations().size();
            int predators = world.getPredatorLocations().size();
            int berries = world.getBerryLocations().size();
            if(cnt % 10 == 0) {
                System.out.println("Iteration: " + cnt);
                System.out.println("Preys: " + preys);
                System.out.println("Predators: " + predators);
                System.out.println("Berries: " + berries);
                System.out.println();
            }
            cnt++;

        }
    }
}
