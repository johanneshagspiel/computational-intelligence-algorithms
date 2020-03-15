import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.List;
import java.util.Scanner;

/**
 * Class that holds all the maze data. This means the pheromones, the open and blocked tiles in the system as
 * well as the starting and end coordinates.
 */
public class Maze {
    private int width;
    private int length;
    private int[][] walls;
    private double[][] pheromones;
    private Coordinate start;
    private Coordinate end;

    /**
     * Constructor of a maze
     * @param walls int array of tiles accessible (1) and non-accessible (0)
     * @param width width of Maze (horizontal)
     * @param length length of Maze (vertical)
     */
    public Maze(int[][] walls, int width, int length) {
        this.walls = walls;
        this.length = length;
        this.width = width;
        initializePheromones();
    }

    /**
     * Initialize pheromones to a start value.
     */
    private void initializePheromones() {

        pheromones = new double[width][length];

        for(int i = 0; i < width;i++){
            for(int j = 0; j < length;j++){
                if(walls[i][j] == 1){
                    pheromones[i][j] = 1;
                }
                else pheromones[i][j] = 0;
            }
        }
    }

    /**
     * Reset the maze for a new shortest path problem.
     */
    public void reset() {
        initializePheromones();
    }

    /**
     * Update the pheromones along a certain route according to a certain Q
     * @param r The route of the ants
     * @param Q Normalization factor for amount of dropped pheromone
     */
    public void addPheromoneRoute(Route r, double Q) {

        Coordinate currentPosition = r.getStart();

        pheromones[currentPosition.getX()][currentPosition.getX()] += (Q/r.getRoute().size());

        for (Direction direction: r.getRoute()) {
            currentPosition.add(direction);
            pheromones[currentPosition.getX()][currentPosition.getX()] += (Q/r.getRoute().size());
        }
    }

    /**
     * Update pheromones for a list of routes
     * @param routes A list of routes
     * @param Q Normalization factor for amount of dropped pheromone
     */
    public void addPheromoneRoutes(List<Route> routes, double Q) {
        for (Route r : routes) {
            addPheromoneRoute(r, Q);
        }
    }

    /**
     * Evaporate pheromone
     * @param rho evaporation factor
     */
    public void evaporate(double rho) {
        for(int i = 0; i < pheromones.length; i++){
            for(int j = 0; j < pheromones[i].length; j++){
                pheromones[i][j] *= (1-rho);
            }
        }
     }

    /**
     * Width getter
     * @return width of the maze
     */
    public int getWidth() {
        return width;
    }

    /**
     * Length getter
     * @return length of the maze
     */
    public int getLength() {
        return length;
    }


    /**
     * Returns a the amount of pheromones on the neighbouring positions (N/S/E/W).
     * @param position The position to check the neighbours of.
     * @return the pheromones of the neighbouring positions.
     */
    public SurroundingPheromone getSurroundingPheromone(Coordinate position) {
        Coordinate checkNorth = new Coordinate(position.getX(), position.getY() + 1);
        Coordinate checkSouth = new Coordinate(position.getX(), position.getY() - 1);
        Coordinate checkEast = new Coordinate(position.getX() - 1, position.getY());
        Coordinate checkWest = new Coordinate(position.getX() + 1, position.getY());
        SurroundingPheromone sp = new SurroundingPheromone(getPheromone(checkNorth),
                getPheromone(checkEast), getPheromone(checkSouth), getPheromone(checkWest));

        return sp;
    }

    /**
     * Pheromone getter for a specific position. If the position is not in bounds returns 0
     * @param pos Position coordinate
     * @return pheromone at point
     */
    private double getPheromone(Coordinate pos) {
        if (pos.getX() >= 0 && pos.getX() < width && pos.getY() >= 0 && pos.getY() < length)
            return pheromones[pos.getX()][pos.getY()];
        else return 0;
    }


    /**
     * Check whether a coordinate lies in the current maze.
     * @param position The position to be checked
     * @return Whether the position is in the current maze
     */
    private boolean inBounds(Coordinate position) {
        return position.xBetween(0, width) && position.yBetween(0, length);
    }

    /**
     * Representation of Maze as defined by the input file format.
     * @return String representation
     */
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(width);
        sb.append(' ');
        sb.append(length);
        sb.append(" \n");
        for (int y = 0; y < length; y++) {
            for (int x = 0; x < width; x++ ) {
                sb.append(walls[x][y]);
                sb.append(' ');
            }
            sb.append("\n");
        }
        return sb.toString();
    }

    /**
     * Method that builds a mze from a file
     * @param filePath Path to the file
     * @return A maze object with pheromones initialized to 0's inaccessible and 1's accessible.
     */
    public static Maze createMaze(String filePath) throws FileNotFoundException {
        Scanner scan = new Scanner(new FileReader(filePath));
        int width = scan.nextInt();
        int length = scan.nextInt();
        int[][] mazeLayout = new int[width][length];
        for (int y = 0; y < length; y++) {
            for (int x = 0; x < width; x++) {
                mazeLayout[x][y] = scan.nextInt();
            }
        }
        scan.close();
        return new Maze(mazeLayout, width, length);
    }
}
