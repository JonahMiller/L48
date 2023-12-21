package PyInterface;

import simulator.ReproductionType;
import simulator.World;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class Callable {
    public static void main(String[] args) {

        ReproductionType reproductionType = ReproductionType.valueOf(args[0]);
        String[] intArgs = Arrays.copyOfRange(args, 1, 6);
        String[] doubleArgs = Arrays.copyOfRange(args, 6, args.length);
        List<Integer> intWorldArgs = Arrays.stream(intArgs).mapToInt(Integer::parseInt).boxed()
                                           .collect(Collectors.toList());
        List<Double> doubleWorldArgs = Arrays.stream(doubleArgs).mapToDouble(Double::parseDouble).boxed()
                                             .collect(Collectors.toList());
        int stepLimit = intWorldArgs.get(0);
        int startingPreyCount = intWorldArgs.get(1);
        int startingPredatorCount = intWorldArgs.get(2);
        int startingBerryCount = intWorldArgs.get(3);
        int maxFoodCount = intWorldArgs.get(4);

        double berrySustenance = doubleWorldArgs.get(0);
        double preySustenance = doubleWorldArgs.get(1);
        double preyStartingEnergy = doubleWorldArgs.get(2);
        double predatorStartingEnergy = doubleWorldArgs.get(3);
        double preyStarvation = doubleWorldArgs.get(4);
        double predatorStarvation = doubleWorldArgs.get(5);
        double preyStepEnergy = doubleWorldArgs.get(6);
        double predatorStepEnergy = doubleWorldArgs.get(7);
        double eatingRadius = doubleWorldArgs.get(8);
        double predatorEatingProbability = doubleWorldArgs.get(9);
        double preyReproductionEnergyThreshold = doubleWorldArgs.get(10);
        double predatorReproductionEnergyThreshold = doubleWorldArgs.get(11);
        double preyReproductionProbability = doubleWorldArgs.get(12);
        double predatorReproductionProbability = doubleWorldArgs.get(13);
        double speed = doubleWorldArgs.get(14);

        double preySpawnRate = doubleWorldArgs.get(15);
        double predatorSpawnRate = doubleWorldArgs.get(16);
        double foodSpawnRate = doubleWorldArgs.get(17);

        double timestep = doubleWorldArgs.get(18);


        World world = new World(reproductionType,
                                startingPreyCount,
                                startingPredatorCount,
                                startingBerryCount,
                                maxFoodCount,
                                berrySustenance,
                                preySustenance,
                                preyStartingEnergy,
                                predatorStartingEnergy,
                                preyStarvation,
                                predatorStarvation,
                                preyStepEnergy,
                                predatorStepEnergy,
                                eatingRadius,
                                predatorEatingProbability,
                                preyReproductionEnergyThreshold,
                                predatorReproductionEnergyThreshold,
                                preyReproductionProbability,
                                predatorReproductionProbability,
                                speed,
                                preySpawnRate,
                                predatorSpawnRate,
                                foodSpawnRate);
        int cnt = 0;
        while(true) {
            try {
                world.advanceTimeBy(timestep);
            } catch(Exception e) {
                System.out.println(e);
            }
            int preys = world.getPreyLocations().size();
            int predators = world.getPredatorLocations().size();
            int berries = world.getBerryLocations().size();

            System.out.printf("%d %d %d %d%n", cnt, preys, predators, berries);

            if(cnt == stepLimit) {
                break;
            }
            cnt++;

        }
    }
}
