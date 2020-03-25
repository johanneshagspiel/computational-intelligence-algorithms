import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

public class Perceptron {
    double[] weightArray;
    double threshold;

    Perceptron(int size)
    {
        weightArray = new double[size];
        for (int i = 0; i < size; i++) {
            weightArray[i] = ThreadLocalRandom.current().nextDouble(-0.5, 0.5);
        }

        threshold = ThreadLocalRandom.current().nextDouble(-0.5, 0.5);
    }

    public int activation(int[] inputArray)
    {
        assert inputArray.length == this.weightArray.length;

        double tempResult = 0;

        for (int i = 0; i < weightArray.length; i++) {
            tempResult += weightArray[i]*inputArray[i];
        }

        tempResult -= this.threshold;

        if (tempResult >= 0)
        {
            return 1;
        }
        else {
            return 0;
        }
    }

    public void weightTraining(int[] inputArray, int result, int desiredResult, double alpha)
    {
        assert inputArray.length == this.weightArray.length;

        int error = desiredResult - result;

        if (error == 0)
        {
            return;
        }

        for (int i = 0; i < this.weightArray.length; i++) {
            weightArray[i] += alpha*error*inputArray[i];
        }
    }

    public void run(int[][] inputArray, int[] desiredResultArray, int epoch, double alpha)
    {
        assert this.weightArray.length == desiredResultArray.length;
        assert this.weightArray.length == inputArray[0].length;

        for (int i = 0; i < epoch; i++) {

            for (int iteration = 0; iteration < inputArray.length; iteration++) {

                int result = activation(inputArray[iteration]);
                int error = desiredResultArray[iteration] - result;

                weightTraining(inputArray[iteration], result, desiredResultArray[iteration], alpha);

                System.out.println("Epoch " + (i+1) + ", Iteration " + (iteration+1) + " Error " + error);

            }

        }

    }
}
