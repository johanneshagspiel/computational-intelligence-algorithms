package tudelft.rl.mysolution;

import java.util.ArrayList;

import tudelft.rl.Action;
import tudelft.rl.QLearning;
import tudelft.rl.State;

public class MyQLearning extends QLearning {

	@Override
	public void updateQ(State s, Action a, double r, State s_next, ArrayList<Action> possibleActions, double alfa, double gamma) {
		double maxval = Integer.MIN_VALUE;
		for (Action act : possibleActions) {
			maxval = Double.max(maxval, getQ(s_next, act));
		}
		double val = getQ(s, a) + alfa * (r + gamma * maxval - getQ(s, a));
//		System.out.println(s + " (" + s.type + ") " + a.id + " => " + s_next + " (" + s_next.type + ") ");
		setQ(s, a, val);
	}

}
