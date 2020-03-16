import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;

/**
 * Class representing the first assignment. Finds shortest path between two points in a maze according to a specific
 * path specification.
 */
public class AntColonyOptimization {
	
	private int antsPerGen;
    private int generations;
    private double Q;
    private double evaporation;
    private Maze maze;

    /**
     * Constructs a new optimization object using ants.
     * @param maze the maze .
     * @param antsPerGen the amount of ants per generation.
     * @param generations the amount of generations.
     * @param Q normalization factor for the amount of dropped pheromone
     * @param evaporation the evaporation factor.
     */
    public AntColonyOptimization(Maze maze, int antsPerGen, int generations, double Q, double evaporation) {
        this.maze = maze;
        this.antsPerGen = antsPerGen;
        this.generations = generations;
        this.Q = Q;
        this.evaporation = evaporation;
    }

    /**
     * Loop that starts the shortest path process
     * @param spec Spefication of the route we wish to optimize
     * @return ACO optimized route
     */
    public Route findShortestRoute(PathSpecification spec) {
        maze.reset();

        ArrayList<Route> routes = null;
        for (int i = 0; i < this.generations; i++) {
            routes = new ArrayList<>();
            Ant[] antArray = new Ant[this.antsPerGen];

            for(int j = 0; j < antArray.length; j++) {
                antArray[j] = new Ant(maze, spec);
                Route r = antArray[j].findRoute();
                if (r != null)
                    routes.add(r);
            }

            maze.evaporate(this.evaporation);
            maze.addPheromoneRoutes(routes, this.Q);
        }

        int minlen = Integer.MAX_VALUE;
        Route res = null;

        for (Route r : routes) {
            if (r.size() < minlen) {
                minlen = r.size();
                res = r;
            }
        }
        return res;
    }

    /**
     * Driver function for Assignment 1
     */
    public static void main(String[] args) throws FileNotFoundException {
    	//parameters
    	int gen = 1000;
        int noGen = 1000;
        double Q = 1600;
        double evap = 0.025;
        
        //construct the optimization objects
        Maze maze = Maze.createMaze("./data/medium maze.txt");
        PathSpecification spec = PathSpecification.readCoordinates("./data/medium coordinates.txt");
        AntColonyOptimization aco = new AntColonyOptimization(maze, gen, noGen, Q, evap);
        
        //save starting time
        long startTime = System.currentTimeMillis();
        
        //run optimization
        Route shortestRoute = aco.findShortestRoute(spec);
        
        //print time taken
        System.out.println("Time taken: " + ((System.currentTimeMillis() - startTime) / 1000.0));
        
        //save solution
        shortestRoute.writeToFile("./data/medium_solution.txt");
        
        //print route size
        System.out.println("Route size: " + shortestRoute.size());
    }
}
