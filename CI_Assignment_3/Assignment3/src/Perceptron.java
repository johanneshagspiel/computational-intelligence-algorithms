import ActivationFunctions.ActivationFunction;

import java.util.concurrent.ThreadLocalRandom;

public class Perceptron {
    double[] weightArray;
    double threshold;
    ActivationFunction activationFunction;

    Perceptron(int size, ActivationFunction activationFunction)
    {
        weightArray = new double[size];
        for (int i = 0; i < size; i++) {
            weightArray[i] = ThreadLocalRandom.current().nextDouble(-0.5, 0.5);
        }

        threshold = ThreadLocalRandom.current().nextDouble(-0.5, 0.5);
        this.activationFunction = activationFunction;

//        threshold = 0.19999999999999998;
//        weightArray[0] = 0.3;
//        weightArray[1] = -0.1;
    }

    public double activation(int[] inputArray)
    {
        assert inputArray.length == this.weightArray.length;

        double tempResult = 0;

        for (int i = 0; i < weightArray.length; i++) {
            tempResult += weightArray[i]*inputArray[i];
        }

        return activationFunction.evaluate(tempResult, threshold);
    }

    public void weightTraining(int[] inputArray, double result, int desiredResult, double alpha)
    {
        assert inputArray.length == this.weightArray.length;

        double error = desiredResult - result;

        if (error == 0)
        {
            return;
        }

        for (int i = 0; i < this.weightArray.length; i++) {
                weightArray[i] += alpha*error*inputArray[i];
        }

        threshold -= alpha*error;
    }

    public void run(int[][] inputArray, int[] desiredResultArray, int epoch, double alpha)
    {
        assert this.weightArray.length == desiredResultArray.length;
        assert this.weightArray.length == inputArray[0].length;

        for (int i = 0; i < epoch; i++) {

            System.out.println("Epoch " + (i+1));
            double averageError = 0;

            for (int iteration = 0; iteration < inputArray.length; iteration++) {

                double result = activation(inputArray[iteration]);
                double error = desiredResultArray[iteration] - result;
                averageError += error;

                weightTraining(inputArray[iteration], result, desiredResultArray[iteration], alpha);

                System.out.println("Iteration " + (iteration+1) + " Error " + error);

            }

            averageError /= epoch;

            System.out.println("Epoch " + (i+1) + " Average Error " + averageError);

        }

    }

    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    public ActivationFunction getActivationFunction() {
        return this.activationFunction;
    }
}
