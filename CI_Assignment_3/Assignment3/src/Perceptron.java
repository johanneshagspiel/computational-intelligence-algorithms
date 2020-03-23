public class Perceptron {
    double[] weightArray;

    public int calc(double[] inputArray)
    {
        assert inputArray.length == this.weightArray.length;

        int result = 0;
        double tempResult = 0;

        for (int i = 0; i < weightArray.length; i++) {
            tempResult = weightArray[i]*inputArray[i];
        }

    }
}
