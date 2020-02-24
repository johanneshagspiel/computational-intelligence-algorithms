package tudelft.rl.mysolution;

import tudelft.rl.*;
import java.util.*;

public class MyEGreedy extends EGreedy {

	@Override
	public Action getRandomAction(Agent r, Maze m) {
		ArrayList<Action> actionList = m.getValidActions(r);
		Random random = new Random();
		Action tempAction = actionList.get(random.nextInt(actionList.size()));
		return tempAction;
	}

	@Override
	public Action getBestAction(Agent r, Maze m, QLearning q) {

		State currentState = r.getState(m);
		ArrayList<Action> actionList = m.getValidActions(r);
		double maxQ = - Double.MAX_VALUE;
		ArrayList<Action> tempActionList = new ArrayList<>();

		for(Action action : actionList) {
			double tempResult = q.getQ(currentState, action);
			if(tempResult > maxQ)
			{
				maxQ = tempResult;
				tempActionList = new ArrayList<>();
				tempActionList.add(action);
			}
			else if (tempResult == maxQ)
			{
				tempActionList.add(action);
			}
		}
		Random random = new Random();
		Action tempAction = tempActionList.get(random.nextInt(tempActionList.size()));
		return tempAction;
	}

	@Override
	public Action getEGreedyAction(Agent r, Maze m, QLearning q, double epsilon) {
		Random random = new Random();
		Double randomDouble = random.nextDouble();

		if(randomDouble > epsilon)
		{
			return getBestAction(r,m ,q);
		}
		else
		{
			return getRandomAction(r, m);
		}
	}
}
