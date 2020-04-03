package ActivationFunctions;

public class HyberbolicTangent implements ActivationFunction {
    @Override
    public double evaluate(double result, double threshold) {
        return ((Math.exp(result)-Math.exp((-1)*result)))/((Math.exp(result)+Math.exp((-1)*result)));
    }

    @Override
    public double getDerivative(double result, double threshold) {
        return 1- Math.pow(evaluate(result,threshold), 2.0);
    }
}
