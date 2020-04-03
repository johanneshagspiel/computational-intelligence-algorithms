package ActivationFunctions;

public class SignFunction implements ActivationFunction {
    @Override
    public double evaluate(double result, double threshold) {

        if (result >= threshold)
        {
            return 1.0;
        }
        else {
            return -1.0;
        }
    }

    @Override
    public double getDerivative(double result, double threshold) {
        return 0;
    }
}
