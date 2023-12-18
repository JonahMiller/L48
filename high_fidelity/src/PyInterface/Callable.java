package PyInterface;

import simulator.ReproductionType;
import simulator.World;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class Callable {
    public static void main(String[] args) {

        ReproductionType reproductionType = ReproductionType.valueOf(args[0]);
        String[] intArgs = Arrays.copyOfRange(args, 1, args.length);
        List<Integer> worldArgs = Arrays.stream(intArgs).mapToInt(Integer::parseInt).boxed().collect(Collectors.toList());

        World world = new World(reproductionType, worldArgs.get(0), worldArgs.get(1), worldArgs.get(2));
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

            if(cnt == 1000) {
                break;
            }
            cnt++;

        }
    }
}
