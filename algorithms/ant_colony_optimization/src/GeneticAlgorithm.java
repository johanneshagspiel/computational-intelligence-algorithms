import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

/**
 * TSP problem solver using genetic algorithms.
 */
public class GeneticAlgorithm {

    private int generations;
    private int popSize;

    /**
     * Constructs a new 'genetic algorithm' object.
     * @param generations the amount of generations.
     * @param popSize the population size.
     */
    public GeneticAlgorithm(int generations, int popSize) {
        this.generations = generations;
        this.popSize = popSize;
    }


    /**
     * Knuth-Yates shuffle, reordering a array randomly
     * @param chromosome array to shuffle.
     */
    private void shuffle(int[] chromosome) {
        int n = chromosome.length;
        for (int i = 0; i < n; i++) {
            int r = i + (int) (Math.random() * (n - i));
            int swap = chromosome[r];
            chromosome[r] = chromosome[i];
            chromosome[i] = swap;
        }
    }

    private int[] crossOver(int[] parent1, int[] parent2) {

        assert(parent1.length == parent2.length);

        ArrayList<Integer> childTemp= new ArrayList<Integer>();
        int[] child = new int[parent1.length];

        Random random = new Random();
        int geneSection1 = random.nextInt(parent1.length);
        int geneSection2 = random.nextInt(parent1.length);
        int start = Math.min(geneSection1,geneSection2);
        int end = Math.max(geneSection1,geneSection2);

        for(int i = start; i <= end; i++)
        {
            childTemp.add(parent1[i]);
        }

        for(int i = 0; i <= parent2.length; i++)
        {
            if(!childTemp.contains(parent2[i]))
            {
                childTemp.add(parent2[i]);
            }
        }

        for (int i = 0; i < childTemp.size(); i++) {
            child[i] = childTemp.get(i);
        }

        return child;
    }

    private void mutate(int[] route, double probMutation) {

        Random random = new Random();
        double tempResult = random.nextDouble();

        if(tempResult >= probMutation)
        {
            int geneSection1 = random.nextInt(route.length);
            int geneSection2 = random.nextInt(route.length);
            int start = Math.min(geneSection1,geneSection2);
            int end = Math.max(geneSection1,geneSection2);

            int tempGene1 = route[start];
            route[start] = route[end];
            route[end] = tempGene1;
        }
    }


    /**
     * This method should solve the TSP. 
     * @param pd the TSP data.
     * @return the optimized product sequence.
     */
    public int[] solveTSP(TSPData pd) {
        return new int[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
    }

    /**
     * Assignment 2.b
     */
    public static void main(String[] args) throws IOException, ClassNotFoundException {
    	//parameters
    	int populationSize = 20;
        int generations = 20;
        String persistFile = "./data/productMatrixDist";
        
        //setup optimization
        TSPData tspData = TSPData.readFromFile(persistFile);
        GeneticAlgorithm ga = new GeneticAlgorithm(generations, populationSize);
        
        //run optimzation and write to file
        int[] solution = ga.solveTSP(tspData);
        tspData.writeActionFile(solution, "./data/TSP_solution.txt");
    }
}
