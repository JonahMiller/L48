package simulator;

import UI.UIFrame;

import javax.swing.*;

public class Main {
    public static void main(String[] args) {
        World world = new World(10,10,10);
        UIFrame frame = new UIFrame(world);
        frame.setVisible(true);
        while(true) {
            try {
                Thread.sleep(20);
                world.advanceTimeBy(1);
                frame.repaint();
            } catch (Exception e) {
                System.out.println(e);
            }
        }
    }
}
