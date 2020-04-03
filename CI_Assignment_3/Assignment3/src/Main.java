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
        System.out.println(ml.run(epoch, alpha, batchedInput, batchedInputLabels, beta, false));
        System.out.println(ml.test(testArray, desiredTestResult));
        
    }
}
