
package ActivationFunctions;

public class SigmoidFunction implements ActivationFunction {
    @Override
    public double evaluate(double result, double threshold) {

        return (1 / (1 + Math.exp((-1)*result)));
    }

    @Override
    public double getDerivative(double result, double threshold) {
        return 0;
    }
}
