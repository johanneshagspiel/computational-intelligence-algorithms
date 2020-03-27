import ActivationFunctions.SigmoidFunction;
import ActivationFunctions.StepFunction;

public class Main {

    public static void main(String[] args) {

        double[][] inputArray = new double[4][2];
        inputArray[0][0] = 0;
        inputArray[1][0] = 0;
        inputArray[2][0] = 1;
        inputArray[3][0] = 1;

        inputArray[0][1] = 0;
        inputArray[1][1] = 1;
        inputArray[2][1] = 0;
        inputArray[3][1] = 1;

        int[] desiredResultArray = new int[4];
        desiredResultArray[0] = 0;
        desiredResultArray[1] = 1;
        desiredResultArray[2] = 1;
        desiredResultArray[3] = 0;

        int epoch = 100;
        double alpha = 0.2;

//	    Perceptron test = new Perceptron(inputArray[0].length, new StepFunction());
//        test.run(inputArray, desiredResultArray, epoch, alpha);
        MultiLayer ml = new MultiLayer(1,2,2,1, new SigmoidFunction());

        for (int i = 0; i < epoch; i++) {
            double epocherror = 0;
            for (int j = 0; j < inputArray.length; j++) {
                double[][] res = ml.process(inputArray[j]);
                double totalerror = 0;
                for (int k = 0; k < res[res.length - 1].length; k++) totalerror += Math.abs(res[res.length - 1][k] - desiredResultArray[k]);
                System.out.println("Average error of iteration " + j + ": " + totalerror/res.length);
                epocherror += totalerror/res.length;
                ml.backPropagate(res, desiredResultArray, alpha);
            }
            System.out.println("Total error of epoch " + i + ": " + epocherror/inputArray.length);
        }

    }
}
