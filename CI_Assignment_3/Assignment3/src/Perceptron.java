import ActivationFunctions.ActivationFunction;

import java.util.concurrent.ThreadLocalRandom;

/**
 * The Perceptron class.
 */
public class Perceptron {
    /**
     * The Weight array.
     */
    double[] weightArray;
    /**
     * The Threshold that determines to which class an input vector belongs to.
     */
    double threshold;
    /**
     * The Activation function used to classify inputs. We implemented the strategy design pattern
     * to enable the easy usage of multiple different activation functions
     */
    ActivationFunction activationFunction;

    /**
     * Instantiates a new Perceptron.
     *
     * @param size               the size of the weight array
     * @param activationFunction the activation function
     */
    Perceptron(int size, ActivationFunction activationFunction)
    {
        weightArray = new double[size];

        //both the treshold and the weights are initialized to random values in the range -0.5 to 0.5
        for (int i = 0; i < size; i++) {
            weightArray[i] = ThreadLocalRandom.current().nextDouble(-0.5, 0.5);
        }

        threshold = ThreadLocalRandom.current().nextDouble(-0.5, 0.5);
        this.activationFunction = activationFunction;
    }

    /**
     * Activation method whose result comes from multiplying the input vector with the weight array and then classifying that
     * outcome based on the activation function.
     *
     * @param inputArray the input array
     * @return the class the input array belongs to
     */
    public double activation(double[] inputArray)
    {
        //first we multiply the input array with the weights
        double tempResult = preprocess(inputArray);

        //then we evaluate the result based on the activation function and the threshold
        return activationFunction.evaluate(tempResult, threshold);
    }

    /**
     * Helper method for multiplying the inputArray with the weights.
     *
     * @param inputArray the input array
     * @return the result of the multiplication
     */
    public double preprocess(double[] inputArray) {
        assert inputArray.length == this.weightArray.length;

        double tempResult = 0;

        for (int i = 0; i < weightArray.length; i++) {
            tempResult += weightArray[i]*inputArray[i];
        }
        return tempResult;
    }

    /**
     * Method to update the weights of the perceptron based on the calculated result, the learning rate and the actual result
     *
     * @param inputArray    the input array
     * @param result        the result
     * @param desiredResult the desired result
     * @param alpha         the learning rate
     */
    public void weightTraining(double[] inputArray, double result, int desiredResult, double alpha)
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

    /**
     * Method to run the perceptron for a certain number of epochs.
     *
     * @param inputArray         the input array
     * @param desiredResultArray the desired result array
     * @param epoch              the epoch
     * @param alpha              the learning rate
     */
    public void run(double[][] inputArray, int[] desiredResultArray, int epoch, double alpha)
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

    /**
     * Sets activation function.
     *
     * @param activationFunction the activation function
     */
    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    /**
     * Gets activation function.
     *
     * @return the activation function
     */
    public ActivationFunction getActivationFunction() {
        return this.activationFunction;
    }
}
