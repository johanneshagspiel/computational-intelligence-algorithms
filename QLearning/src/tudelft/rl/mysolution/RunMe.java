package tudelft.rl.mysolution;

import java.io.File;
import java.util.Map;
import java.util.Random;

import tudelft.rl.*;

public class RunMe {

	public static double alfa = 0.7;
	public static double gamma = 0.9;
//	public static double epsilon = 0.1;

	public static void main(String[] args) {

        double epsilon = 0.99999;

        //load the maze
        //TODO replace this with the location to your maze on your file system
        Maze maze = new Maze(new File("QLearning\\data\\easy_maze.txt"));

        //Set the reward at the bottom right to 10
        maze.setR(maze.getState(9, 9), 10);
        maze.setR(maze.getState(9, 0), 5);

        //create a robot at starting and reset location (0,0) (top left)
        Agent robot = new Agent(0, 0);

        //make a selection object (you need to implement the methods in this class)
        EGreedy selection = new MyEGreedy();

        //make a Qlearning object (you need to implement the methods in this class)
        QLearning learn = new MyQLearning();
        boolean stop = false;
        int numberOfTotalSteps = 0;
        //keep learning until you decide to stop
        while (!stop) {
            //TODO implement the action selection and learning cycle
            numberOfTotalSteps++;
            if (numberOfTotalSteps > 300000) {
                stop = true;
            } else {
                Action action = selection.getEGreedyAction(robot, maze, learn, epsilon /*> 0.7 ? 1 : epsilon*/);
                State prevstate = robot.getState(maze);
                robot.doAction(action, maze);
                learn.updateQ(prevstate, action, maze.getR(robot.getState(maze)), robot.getState(maze), maze.getValidActions(robot), alfa, gamma);
                if (robot.checkForReset()) epsilon *= epsilon;
            }
        }
    }

}
