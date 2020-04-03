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

        //we first set the hyperparameters. These were determined via both
        int epoch = 500;
        double alpha = 0.1;
        double beta = 0.4;

        //then we create the first data object based on the features.txt and targets.txt files
        //this data object will be used to train the mlp
        DataObject data;
        try {
            data = new DataObject("features.txt", "targets.txt", 0.1, 0.1, 100);
        } catch (IOException e) {
            e.printStackTrace(); //no data? Terminate
            return;
        }

        //now we create the required arrays to input into the mlp, namely batchedInput with the features and batchedInputLabels
        //with the associated classes to train the mlp
        double[][][] batchedInput = data.batches;;
        int[][][] batchedInputLabels = data.batchlabels;

        //we create the mlp based on a sigmoid activation function, 1 layer of hidden neurons and 8 neurons per layer
        //these parameters were determined based on extensive testing and experimentation
        MultiLayer mlp = new MultiLayer(1, 8, batchedInput[0][0].length, batchedInputLabels[0][0].length, new SigmoidFunction());

        //next, we train the mlp based on the initially set hyperparameters, feature and label arrays.
        //the usage of heuristics is turned off as they greatly decrease training time but substantially worsen the accuracy of the mlp
        mlp.run(epoch, alpha, batchedInput, batchedInputLabels, beta, false);

        //then, we create the second data object based on the unknown.txt file which we will use to predict the classes
        DataObject data2;
        try {
            data2 = new DataObject("unknown.txt" );
        } catch (IOException e) {
            e.printStackTrace(); //no data? Terminate
            return;
        }

        //again we need to turn this data object into suitable input arrays
        double[][] unknownArray = data2.inputFeatures;

        //finally, we generate a csv file with the predicted classes
        mlp.output(unknownArray);
    }
}
