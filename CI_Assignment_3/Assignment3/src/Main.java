import ActivationFunctions.SigmoidFunction;
import ActivationFunctions.StepFunction;

import java.io.File;
import java.io.IOException;
import java.util.Scanner;

public class Main {

    public static void main(String[] args) {

        //here, we declare the input and desired output of the algorithm
//        double[][] inputArray = new double[4][2];
//        inputArray[0][0] = 0;
//        inputArray[1][0] = 0;
//        inputArray[2][0] = 1;
//        inputArray[3][0] = 1;
//
//        inputArray[0][1] = 0;
//        inputArray[1][1] = 1;
//        inputArray[2][1] = 0;
//        inputArray[3][1] = 1;
//
//        int[][] desiredResultArray = new int[4][1];
//        desiredResultArray[0][0] = 0;
//        desiredResultArray[1][0] = 1;
//        desiredResultArray[2][0] = 1;
//        desiredResultArray[3][0] = 1;

        double[][] inputArray = new double[7854][10];
        int[][] desiredResultArray = new int[7854][7];

        try {
            Scanner sc = new Scanner(new File("features.txt"));
            Scanner sc1;
            for (int i = 0; i < inputArray.length; i++) {
                sc1 = new Scanner(sc.nextLine());
                sc1.useDelimiter(",");
                for (int j = 0; j < inputArray[i].length; j++) inputArray[i][j] = sc1.nextDouble();
            }

            sc = new Scanner(new File("targets.txt"));
            for (int i = 0; i < desiredResultArray.length; i++) {
                desiredResultArray[i][sc.nextInt() - 1] = 1;
            }
        } catch (IOException e){
            System.out.println("Error reading file: " + e);
            e.printStackTrace();
            return;
        }

        //we also set the hyperparameters
        int epoch = 100;
        double alpha = 0.2;

        //we now instantiate the object with the right data, and run it

//	    Perceptron test = new Perceptron(inputArray[0].length, new StepFunction());
//        test.run(inputArray, desiredResultArray, epoch, alpha);
        MultiLayer ml = new MultiLayer(1, 8, inputArray[0].length, desiredResultArray[0].length, new SigmoidFunction());
        ml.run(epoch, alpha, inputArray, desiredResultArray);

    }
}
