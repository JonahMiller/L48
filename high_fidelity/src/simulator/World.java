package simulator;

import java.util.*;
import java.util.stream.Collectors;

public class World {
    Set<Animal> animals;
    Set<Food> foods;

    private final double foodSpawnRate = 10;

    private final double spontaneousReproductionRate = 0.1;

    private final double preySpawnRate = 0.05;
    private final double predatorSpawnRate = 0.05;

    public static final double minX = 0;
    public static final double maxX = 1000;
    public static final double minY = 0;
    public static final double maxY = 1000;

    private int getFoodCount(double timespan) {
        double foodCountExpectation = timespan * foodSpawnRate;
        int foodCount = (int) Math.floor(foodCountExpectation);
        foodCountExpectation -= foodCount;
        if (Math.random() < foodCountExpectation) {
            foodCount++;
        }
        return foodCount;
    }

    private WorldView getWorldView(Point centre, double radius) {
        Set<Animal> animalsSeen = animals.stream().filter(x -> centre.getDistance(x.getLocation()) <= radius)
                .collect(Collectors.toSet());
        Set<Food> foodsSeen = foods.stream().filter(x -> centre.getDistance(x.getLocation()) <= radius)
                .collect(Collectors.toSet());
        return new WorldView(animalsSeen, foodsSeen);
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

        for (int i = 0; i < foodCount; i++) {
            foods.add(new Berry(getRandomLocation()));
        }
    }

    private void starveAnimals(double timespan) {
        animals.forEach(animal -> animal.starve(timespan));
    }

    private void handleDeaths() {
        animals = animals.stream().filter(Animal::isAlive).collect(Collectors.toSet());
        foods = foods.stream().filter(Food::exists).collect(Collectors.toSet());
    }

    // Animals reproduce based on food
    private void reproduceAnimals() {
        Set<Animal> offsprings = animals.stream().filter(Animal::canReproduce).map(Animal::reproduce)
                .filter(Objects::nonNull).collect(Collectors.toSet());

        animals.addAll(offsprings);

        Set<Food> foodOffsprings = new HashSet<>();
        for (Animal x : offsprings) {
            if (x instanceof Food) {
                foodOffsprings.add((Food) x);
            }
        }

        foods.addAll(foodOffsprings);
    }

    private double sampleExponentialVariable(double lambda) {
        // We do 1-[0,1) as Math.random() might return 0 but not 1
        return -Math.log(1 - Math.random()) / lambda;
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

    // Animals reproduce spontaneously regardless of food level
    private void reproduceAnimals2(double timespan) {
        // TODO: Can this be optimised? Does it have to be?
        Set<Animal> offsprings = animals.stream().map(x -> spontaneouslyReproduceAnimal(x, timespan))
                .reduce(new HashSet<>(), (x, y) -> {
                    x.addAll(y);
                    return y;
                });

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
            if (Math.random() * (preySpawnRate + predatorSpawnRate) < preySpawnRate) { // We spawn a prey
                Prey prey = new Prey(getRandomLocation());
                this.animals.add(prey);
                this.foods.add(prey);
            } else { // We spawn a predator
                Predator predator = new Predator(getRandomLocation());
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

        reproduceAnimals();

        spawnAnimalsSpontaneously(timespan);

    }

    private Point getRandomLocation() {
        double x = Math.random() * (maxX - minX) + minX;
        double y = Math.random() * (maxY - minY) + minY;
        return new Point(x, y);
    }

    public World(int preyCount, int predatorCount, int berryCount) {
        this.animals = new HashSet<Animal>();
        this.foods = new HashSet<Food>();

        for (int i = 0; i < preyCount; i++) {
            Prey prey = new Prey(getRandomLocation());
            this.animals.add(prey);
            this.foods.add(prey);
        }

        for (int i = 0; i < predatorCount; i++) {
            Predator predator = new Predator(getRandomLocation());
            this.animals.add(predator);
        }

        for (int i = 0; i < berryCount; i++) {
            Berry berry = new Berry(getRandomLocation());
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
