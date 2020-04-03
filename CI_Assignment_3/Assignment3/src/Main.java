import ActivationFunctions.HyberbolicTangent;
import ActivationFunctions.SigmoidFunction;

import javax.xml.crypto.Data;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) throws IOException {

        //we first set the hyperparameters
        int epoch = 500;
        double alpha = 0.1;
        double beta = 0.4;

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

        MultiLayer ml = new MultiLayer(1, 8, batchedInput[0][0].length, batchedInputLabels[0][0].length, new SigmoidFunction());
        ml.run(epoch, alpha, batchedInput, batchedInputLabels, beta, false);
        int[][] test = ml.confusionMatrix(testArray, desiredTestResult);

        for (int i = 0; i < 7; i++) {
            for (int j = 0; j < 7; j++) {
                //System.out.println("Actual class " + (i + 1) + " and predicted class " + (j + 1) + " has cases: " +  test[i][j]);
                System.out.println((i + 1) + " " + (j + 1) + " " + test[i][j]);
            }
        }
    }
}
