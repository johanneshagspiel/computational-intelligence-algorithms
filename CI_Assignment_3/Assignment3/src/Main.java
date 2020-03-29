import ActivationFunctions.SigmoidFunction;

import java.io.IOException;

public class Main {

    public static void main(String[] args) {

        //we first set the hyperparameters
        int epoch = 100;
        double alpha = 0.2;

        //then, we declare the input and desired output of the algorithm
        double[][] inputArray, testArray;
        int[][] desiredResultArray, desiredTestResult;
        DataObject data;
        try {
            data = new DataObject("features.txt", "targets.txt", 0.1, 0.1);
        } catch (IOException e) {
            e.printStackTrace(); //no data? Terminate
            return;
        }

        inputArray = data.trainFeatures;
        desiredResultArray = data.trainLabels;
        testArray = data.testFeatures;
        desiredTestResult = data.testLabels;

        //we now instantiate the object with the right data, and run it

        MultiLayer ml = new MultiLayer(1, 8, inputArray[0].length, desiredResultArray[0].length, new SigmoidFunction());
        ml.run(epoch, alpha, inputArray, desiredResultArray);

        //after training, run a test using test data to see how we did
        ml.test(testArray, desiredTestResult);

    }
}
