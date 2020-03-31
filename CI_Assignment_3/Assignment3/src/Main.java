import ActivationFunctions.HyberbolicTangent;
import ActivationFunctions.SigmoidFunction;

import java.io.IOException;

public class Main {

    public static void main(String[] args) {

        //we first set the hyperparameters
        int epoch = 500;
        double alpha = 0.2;
        double beta = 0.9;

        //then, we declare the input and desired output of the algorithm
        double[][][] batchedInput;
        int[][][] batchedInputLabels;
        double[][] testArray;
        int[][] desiredTestResult;
        DataObject data;
        try {
            data = new DataObject("features.txt", "targets.txt", 0.1, 0.1, 100);
        } catch (IOException e) {
            e.printStackTrace(); //no data? Terminate
            return;
        }

        batchedInput = data.batches;
        batchedInputLabels = data.batchlabels;
        testArray = data.testFeatures;
        desiredTestResult = data.testLabels;

        //we now instantiate the object with the right data, and run it

        MultiLayer ml = new MultiLayer(1, 8, batchedInput[0][0].length, batchedInputLabels[0][0].length, new HyberbolicTangent());

        int betaYes = 0;

        MultiLayer[] mlHyperArray = new MultiLayer[100];

        for (int i = 0; i < 100; i++) {
            mlHyperArray[i] = new MultiLayer(1, 8, batchedInput[0][0].length, batchedInputLabels[0][0].length, new HyberbolicTangent());
        }

        for (int i = 0; i < 100; i++) {
            betaYes += mlHyperArray[i].getEpochsToConvergence(beta, alpha, batchedInput, batchedInputLabels, Math.pow(10, -1), true);
        }
        betaYes /= 100;

        int betaNo = 0;

        MultiLayer[] mlSigmoidArray = new MultiLayer[100];

        for (int i = 0; i < 100; i++) {
            mlSigmoidArray[i] = new MultiLayer(1, 8, batchedInput[0][0].length, batchedInputLabels[0][0].length, new HyberbolicTangent());
        }

        for (int i = 0; i < 100; i++) {
            betaNo += mlSigmoidArray[i].getEpochsToConvergence(0, alpha, batchedInput, batchedInputLabels, Math.pow(10, -1), true);
        }
        betaNo /= 100;

        System.out.println(betaYes + " " + betaNo);

        //after training, run a test using test data to see how we did
        //ml.test(testArray, desiredTestResult);

    }
}
