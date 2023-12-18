package simulator;

import java.util.*;
import java.util.stream.Collectors;

public class World {
    Set<Animal> animals;
    Set<Food> foods;

    public final static Random rng = new Random(0);

    private final int maxFoodCount;

    private final double berrySustenance;
    private final double preySustenance;
    private final double startingFoodLevel;
    private final double starvationCoefficient;
    private final double eatingRadius;
    private final double reproductionFoodLevel;
    private final double speed;

    private final double spontaneousReproductionRate = 0.1;

    private final double preySpawnRate;
    private final double predatorSpawnRate;
    private final double foodSpawnRate;

    public static final double minX = 0;
    public static final double maxX = 1000;
    public static final double minY = 0;
    public static final double maxY = 1000;

    private ReproductionType reproductionType;

    private int getFoodCount(double timespan) {
        double foodCountExpectation = timespan * foodSpawnRate;
        int foodCount = (int) Math.floor(foodCountExpectation);
        foodCountExpectation -= foodCount;
        if (rng.nextDouble() < foodCountExpectation) {
            foodCount++;
        }
        return foodCount;
    }

    private WorldView getWorldView(Point centre, double radius) {
//        Set<Animal> animalsSeen = animals.stream().filter(x -> centre.getDistance(x.getLocation()) <= radius)
//                .collect(Collectors.toSet());
//        Set<Food> foodsSeen = foods.stream().filter(x -> centre.getDistance(x.getLocation()) <= radius)
//                .collect(Collectors.toSet());
//        return new WorldView(animalsSeen, foodsSeen);
        return new WorldView(animals, foods);
    }

    private void moveAnimals(double timespan) {
        animals.forEach(x -> x.move(getWorldView(x.getLocation(), x.getViewRadius()), timespan));
    }

    private void feedAnimals() {
        for (Animal animal : animals.stream().collect(Collectors.toList())) {
            if (animal.isAlive()) {
                Set<Food> meal = animal.eat(getWorldView(animal.getLocation(), animal.getViewRadius()));
                meal.forEach(Food::consumed);
                foods.removeAll(meal);
            }
        }
    }

    private void spawnFood(double timespan) {
        // TODO: Do we want to cap the total amount of food on the map?
        int foodCount = getFoodCount(timespan);
        if(getBerryLocations().size() > maxFoodCount) {
            foodCount = 0;
        }

        for (int i = 0; i < foodCount; i++) {
            foods.add(new Berry(getRandomLocation(), berrySustenance));
        }
    }

    private void starveAnimals(double timespan) {
        animals.forEach(animal -> animal.starve(timespan));
    }

    private void handleDeaths() {
        animals = animals.stream().filter(Animal::isAlive).collect(Collectors.toSet());
        foods = foods.stream().filter(Food::exists).collect(Collectors.toSet());
    }
    private double sampleExponentialVariable(double lambda) {
        // We do 1-[0,1) as rng.nextDouble() might return 0 but not 1
        return -Math.log(1 - rng.nextDouble()) / lambda;
    }

    private Set<Animal> spontaneouslyReproduceAnimal(Animal animal, double timespan) {
        double nextReproductionTime = sampleExponentialVariable(spontaneousReproductionRate);
        Set<Animal> offsprings = new HashSet<>();
        if (nextReproductionTime <= timespan) {
            Animal offspring = animal.reproduce();
            offsprings.add(offspring);
            // TODO: Do we want these calls?
            offsprings.addAll(spontaneouslyReproduceAnimal(animal, timespan - nextReproductionTime));
            offsprings.addAll(spontaneouslyReproduceAnimal(offspring, timespan - nextReproductionTime));
        }
        return offsprings;
    }

    private void reproduceAnimals(ReproductionType reproductionType, double timespan) {
        Set<Animal> offsprings;
        if(reproductionType == ReproductionType.FOOD) {
            // Animals reproduce based on food level
            offsprings = animals.stream().filter(Animal::canReproduce).map(Animal::reproduce)
                    .filter(Objects::nonNull).collect(Collectors.toSet());
        } else {
            // Animals reproduce spontaneously regardless of food level
            // TODO: Can this be optimised? Does it have to be?
            offsprings = animals.stream().map(x -> spontaneouslyReproduceAnimal(x, timespan))
                    .reduce(new HashSet<>(), (x, y) -> {
                        x.addAll(y);
                        return y;
                    });
        }

        animals.addAll(offsprings);

        Set<Food> foodOffsprings = new HashSet<>();
        for (Animal x : offsprings) {
            if (x instanceof Food) {
                foodOffsprings.add((Food) x);
            }
        }

        foods.addAll(foodOffsprings);
    }

    private void spawnAnimalsSpontaneously(double timespan) {
        double nextSpawnTime = sampleExponentialVariable(preySpawnRate + predatorSpawnRate);
        // Could do recursively but it would be slower
        while (nextSpawnTime <= timespan) {
            if (rng.nextDouble() * (preySpawnRate + predatorSpawnRate) < preySpawnRate) { // We spawn a prey
                Prey prey = new Prey(getRandomLocation(), startingFoodLevel, starvationCoefficient, eatingRadius, reproductionFoodLevel, speed, preySustenance);
                this.animals.add(prey);
                this.foods.add(prey);
            } else { // We spawn a predator
                Predator predator = new Predator(getRandomLocation(), startingFoodLevel, starvationCoefficient, eatingRadius, reproductionFoodLevel, speed);
                this.animals.add(predator);
            }

            timespan -= nextSpawnTime;
            nextSpawnTime = sampleExponentialVariable(preySpawnRate + predatorSpawnRate);
        }
    }

    public void advanceTimeBy(double timespan) {
        moveAnimals(timespan);

        feedAnimals();

        spawnFood(timespan);

        starveAnimals(timespan);

        handleDeaths();

        reproduceAnimals(reproductionType, timespan);

        spawnAnimalsSpontaneously(timespan);

    }

    private Point getRandomLocation() {
        double x = rng.nextDouble() * (maxX - minX) + minX;
        double y = rng.nextDouble() * (maxY - minY) + minY;
        return new Point(x, y);
    }

    public World(ReproductionType reproductionType,
                 int startingPreyCount,
                 int startingPredatorCount,
                 int startingBerryCount,
                 int maxFoodCount,
                 double berrySustenance,
                 double preySustenance,
                 double startingFoodLevel,
                 double starvationCoefficient,
                 double eatingRadius,
                 double reproductionFoodLevel,
                 double speed,
                 double preySpawnRate,
                 double predatorSpawnRate,
                 double foodSpawnRate) {
        this.reproductionType = reproductionType;
        this.maxFoodCount = maxFoodCount;
        this.berrySustenance = berrySustenance;
        this.preySustenance = preySustenance;
        this.startingFoodLevel = startingFoodLevel;
        this.starvationCoefficient = starvationCoefficient;
        this.eatingRadius = eatingRadius;
        this.reproductionFoodLevel = reproductionFoodLevel;
        this.preySpawnRate = preySpawnRate;
        this.predatorSpawnRate = predatorSpawnRate;
        this.foodSpawnRate = foodSpawnRate;
        this.speed = speed;

        this.animals = new HashSet<Animal>();
        this.foods = new HashSet<Food>();

        for (int i = 0; i < startingPreyCount; i++) {
            Prey prey = new Prey(getRandomLocation(), startingFoodLevel, starvationCoefficient, eatingRadius, reproductionFoodLevel, speed, preySustenance);
            this.animals.add(prey);
            this.foods.add(prey);
        }

        for (int i = 0; i < startingPredatorCount; i++) {
            Predator predator = new Predator(getRandomLocation(), startingFoodLevel, starvationCoefficient, eatingRadius, reproductionFoodLevel, speed);
            this.animals.add(predator);
        }

        for (int i = 0; i < startingBerryCount; i++) {
            Berry berry = new Berry(getRandomLocation(), berrySustenance);
            this.foods.add(berry);
        }
    }

    // Maybe include sizes later
    public List<Point> getPreyLocations() {
        List<Point> preyLocations = new ArrayList<>();
        for (Animal animal : animals) {
            if (animal instanceof Prey) {
                preyLocations.add(animal.getLocation());
            }
        }
        return preyLocations;
    }

    // Maybe include sizes later
    public List<Point> getPredatorLocations() {
        List<Point> predatorLocations = new ArrayList<>();
        for (Animal animal : animals) {
            if (animal instanceof Predator) {
                predatorLocations.add(animal.getLocation());
            }
        }
        return predatorLocations;
    }

    // Maybe include sizes later
    public List<Point> getBerryLocations() {
        List<Point> berryLocations = new ArrayList<>();
        for (Food food : foods) {
            if (food instanceof Berry) {
                berryLocations.add(food.getLocation());
            }
        }
        return berryLocations;
    }
}
