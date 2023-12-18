package PyInterface;

import simulator.World;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class Callable {
    public static void main(String[] args) {
        List<Integer> formArgs = Arrays.stream(args).mapToInt(Integer::parseInt).boxed().collect(Collectors.toList());
        World world = new World(formArgs.get(0), formArgs.get(1), formArgs.get(2));
        int cnt = 0;
        while(true) {
            try {
                world.advanceTimeBy(0.5);
            } catch (Exception e) {
                System.out.println(e);
            }
            int preys = world.getPreyLocations().size();
            int predators = world.getPredatorLocations().size();
            int berries = world.getBerryLocations().size();

            System.out.printf("%d %d %d %d%n", cnt, preys, predators, berries);

            if(preys == 0 || cnt>=500) {
                break;
            }
            cnt++;

        }
    }
}
