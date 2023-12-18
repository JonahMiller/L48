package simulator;

import java.util.*;
import java.util.stream.Collectors;

public class World {
    Set<Agent> animals;
    Set<Food> foods;

    private int foodDropRate = 10;

    public static final double minX = 0;
    public static final double maxX = 1000;
    public static final double minY = 0;
    public static final double maxY = 1000;

    private int getFoodCount(double timespan) {
        double foodCountExpectation = timespan * foodDropRate;
        int foodCount = (int) Math.floor(foodCountExpectation);
        foodCountExpectation-= foodCount;
        if(Math.random() < foodCountExpectation) {
            foodCount++;
        }
        return foodCount;
    }

    private WorldView getWorldView(Point centre, double radius) {
        Set<Agent> animalsSeen = animals.stream().filter(x -> centre.getDistance(x.getLocation()) <= radius).collect(Collectors.toSet());
        Set<Food> foodsSeen = foods.stream().filter(x -> centre.getDistance(x.getLocation()) <= radius).collect(Collectors.toSet());
        return new WorldView(animalsSeen, foodsSeen);
    }

    private void moveAnimals(double timespan) {
        animals.forEach(x -> x.move(getWorldView(x.getLocation(), x.getViewRadius()), timespan));
    }


    private void feedAnimals() {
        for (Agent agent : animals.stream().collect(Collectors.toList()))
        {
            if (agent.isAlive())
            {
                Set<Food> meal = agent.eat(getWorldView(agent.getLocation(), agent.getViewRadius()));
                meal.forEach(Food::consumed);
                foods.removeAll(meal);
            }
        }
    }

    private void spawnFood(double timespan) {

        int foodDrop = getFoodCount(timespan);

        for (int i = 0; i < foodDrop; i++)
        {
            foods.add(new Berry(getRandomLocation()));
        }
    }

    private void starveAnimals(double timespan) {
        animals.forEach(animal -> animal.starve(timespan));
    }

    private void handleDeaths() {
        animals = animals.stream().filter(Agent::isAlive).collect(Collectors.toSet());
        foods = foods.stream().filter(Food::exists).collect(Collectors.toSet());
    }

    private void reproduceAnimals() {
        Set<Agent> offsprings = animals.stream().map(Agent::reproduce).filter(Objects::nonNull).collect(Collectors.toSet());

        animals.addAll(offsprings);

        Set<Food> foodOffsprings = new HashSet<>();
        for (Agent x : offsprings) {
            if (x instanceof Food) {
                foodOffsprings.add((Food) x);
            }
        }

        foods.addAll(foodOffsprings);
    }



    public void advanceTimeBy(double timespan) {
        moveAnimals(timespan);

        feedAnimals();

        spawnFood(timespan);

        starveAnimals(timespan);

        handleDeaths();

        reproduceAnimals();


    }

    private Point getRandomLocation() {
        double x = Math.random() * (maxX - minX) + minX;
        double y = Math.random() * (maxY - minY) + minY;
        return new Point(x,y);
    }

    public World(int preyCount, int predatorCount, int berryCount) {
        this.animals = new HashSet<Agent>();
        this.foods = new HashSet<Food>();

        for(int i = 0; i < preyCount; i++) {
            Prey prey = new Prey(getRandomLocation());
            this.animals.add(prey);
            this.foods.add(prey);
        }


        for(int i = 0; i < predatorCount; i++) {
            Predator predator = new Predator(getRandomLocation());
            this.animals.add(predator);
        }


        for(int i = 0; i < berryCount; i++) {
            Berry berry = new Berry(getRandomLocation());
            this.foods.add(berry);
        }
    }

    // Maybe include sizes later
    public List<Point> getPreyLocations() {
        List<Point> preyLocations = new ArrayList<>();
        for(Agent animal : animals) {
            if(animal instanceof Prey) {
                preyLocations.add(animal.getLocation());
            }
        }
        return preyLocations;
    }


    // Maybe include sizes later
    public List<Point> getPredatorLocations() {
        List<Point> predatorLocations = new ArrayList<>();
        for(Agent animal : animals) {
            if(animal instanceof Predator) {
                predatorLocations.add(animal.getLocation());
            }
        }
        return predatorLocations;
    }


    // Maybe include sizes later
    public List<Point> getBerryLocations() {
        List<Point> berryLocations = new ArrayList<>();
        for(Food food : foods) {
            if(food instanceof Berry) {
                berryLocations.add(food.getLocation());
            }
        }
        return berryLocations;
    }
}
