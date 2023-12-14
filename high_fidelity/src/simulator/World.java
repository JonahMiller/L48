package simulator;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class World {
    Set<Agent> animals;
    Set<Food> foods;

    private int foodDropRate = 10;

    public static final double minX = 0;
    public static final double maxX = 1000;
    public static final double minY = 0;
    public static final double maxY = 1000;

    private WorldView getWorldView(Point centre, double radius) {
        Set<Agent> animalsSeen = animals.stream().filter(x -> centre.getDistance(x.getLocation()) <= radius).collect(Collectors.toSet());
        Set<Food> foodsSeen = foods.stream().filter(x -> centre.getDistance(x.getLocation()) <= radius).collect(Collectors.toSet());
        return new WorldView(animalsSeen, foodsSeen);
    }

    public void advanceTimeBy(double timespan) {
        animals.forEach(x -> x.move(getWorldView(x.getLocation(), x.getViewRadius()), timespan));


        for (Agent agent : animals.stream().toList())
        {
            if (agent.isAlive())
            {
                Set<Food> meal = agent.eat(getWorldView(agent.getLocation(), agent.getViewRadius()));
                meal.forEach(Food::consumed);
                foods.removeAll(meal);
            }
        }

        int foodDrop = (Math.random() * timespan < 0.1 ? 1 : 0);

        for (int i = 0; i< foodDrop; i++)
        {
            foods.add(new Berry(getRandomLocation()));
        }

        animals = animals.stream().filter(Agent::isAlive).collect(Collectors.toSet());

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
