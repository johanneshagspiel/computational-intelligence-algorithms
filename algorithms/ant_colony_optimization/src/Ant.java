import java.util.*;

/**
 * Class that represents the ants functionality.
 */
public class Ant {
	
    private Maze maze;
    private Coordinate start;
    private Coordinate end;
    private Coordinate currentPosition;
    private static Random rand;
    private Set<Coordinate> prevs;
    private boolean isFinal = false;

    /**
     * Constructor for ant taking a Maze and PathSpecification.
     * @param maze Maze the ant will be running in.
     * @param spec The path specification consisting of a start coordinate and an end coordinate.
     */
    public Ant(Maze maze, PathSpecification spec) {
        this.maze = maze;
        this.start = spec.getStart();
        this.end = spec.getEnd();
        this.currentPosition = start;
        if (rand == null) {
            rand = new Random();
        }
        prevs = new HashSet<>();
    }

    public Ant(Maze maze, PathSpecification spec, boolean isFinal) {
        this(maze, spec);
        this.isFinal = isFinal;
    }

    /**
     * Method that performs a single run through the maze by the ant.
     * @return The route the ant found through the maze.
     */
    public Route findRoute() {
        return findRoute(30000);
    }

    public Route findRoute(int maxSteps) {
        Route route = new Route(start);
        int steps = 0;

        while(steps++ < maxSteps) {

            SurroundingPheromone sp = maze.getSurroundingPheromone(currentPosition);
            double relevantPheromone;
            boolean DeadEnd;
            List<Direction> options;

            do {
                options = new ArrayList<>(4);
                relevantPheromone = 0;
                for (Direction d : Direction.values()) {

                    if (!prevs.contains(currentPosition.add(d)) && sp.get(d) > 0) {
                        relevantPheromone += sp.get(d);
                        options.add(d);
                    }
                }

                if (relevantPheromone == 0) {
                    //System.out.println(currentPosition + " is dead end");

                    currentPosition = currentPosition.subtract(route.pop());
                    sp = maze.getSurroundingPheromone(currentPosition);
                    DeadEnd = true;
                } else {
                    DeadEnd = false;
                }
            } while (DeadEnd);


            double choice = relevantPheromone * rand.nextDouble();
            double max = 0;
            Direction taken = null;
            for (Direction d : options) {
//                System.out.println(choice + ", " + d + "," + sp.get(d));
                choice -= sp.get(d);
                if (choice < 0 && !isFinal) {
                    taken = d;
//                    System.out.println(choice + "<0, " + taken + "->" + currentPosition.add(taken) + ", " + Arrays.toString(options.toArray()));
                    break;
                }
                if (isFinal) {
                    if (sp.get(d) > max) {
                        max = sp.get(d);
                        taken = d;
                    }
                }
            }


            route.add(taken);
            currentPosition = currentPosition.add(taken);
            prevs.add(currentPosition);
            if (currentPosition.equals(end)) {
                //System.out.println("FOUND");
                return route;
            }
        }
        return null;
    }

}

