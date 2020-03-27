package ActivationFunctions;

public interface ActivationFunction {

    public double evaluate(double result, double threshold);

    public double getDerivative(double result, double threshold);

}
