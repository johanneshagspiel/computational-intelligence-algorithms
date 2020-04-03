
package ActivationFunctions;

public class SigmoidFunction implements ActivationFunction {
    @Override
    public double evaluate(double result, double threshold) {

        return (1 / (1 + Math.exp(threshold - result)));
    }

    @Override
    public double getDerivative(double result, double threshold) {
        return (evaluate(result, threshold))*(1-(evaluate(result, threshold)));
    }
}
