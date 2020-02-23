package tudelft.rl.mysolution;

import java.util.ArrayList;

import tudelft.rl.Action;
import tudelft.rl.QLearning;
import tudelft.rl.State;

public class MyQLearning extends QLearning {

	public double alfa = 0.7;
	public double gamma = 0.9;
	public double epsilon = 0.1;

	@Override
	public void updateQ(State s, Action a, double r, State s_next, ArrayList<Action> possibleActions, double alfa, double gamma) {
		double maxval = Integer.MIN_VALUE;
		for (Action act : possibleActions) {
			maxval = Double.max(maxval, getQ(s_next, act));
		}
		double val = getQ(s, a) + alfa * (r + gamma * maxval - getQ(s, a));
		setQ(s, a, val);
	}

}
