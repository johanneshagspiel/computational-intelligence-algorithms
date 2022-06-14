package ActivationFunctions;

/**
 * The interface Activation function.
 */
public interface ActivationFunction {

    /**
     * Method to determine to which class an input belongs to based on a threshold.
     *
     * @param result    the result
     * @param threshold the threshold
     * @return the double
     */
    public double evaluate(double result, double threshold);

    /**
     * Method to determine the derivative of this perceptron based on the threshold and the result. Needed for the
     * backpropagation algorithm of a mlp.
     *
     * @param result    the result
     * @param threshold the threshold
     * @return the derivative
     */
    public double getDerivative(double result, double threshold);

}
