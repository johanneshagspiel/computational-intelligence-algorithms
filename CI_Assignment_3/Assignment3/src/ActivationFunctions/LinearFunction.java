package ActivationFunctions;

public class LinearFunction implements ActivationFunction {
    @Override
    public double evaluate(double result, double threshold) {

        return result;
    }

    @Override
    public double getDerivative(double result, double threshold) {
        return 1.0;
    }
}
