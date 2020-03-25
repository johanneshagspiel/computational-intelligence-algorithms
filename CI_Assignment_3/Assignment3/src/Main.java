public class Main {

    public static void main(String[] args) {

        int[][] inputArray = new int[4][2];
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
        desiredResultArray[1] = 0;
        desiredResultArray[2] = 0;
        desiredResultArray[3] = 1;

        int epoch = 5;
        double alpha = 0.1;

	    Perceptron test = new Perceptron(inputArray[0].length);

	    test.run(inputArray, desiredResultArray, epoch, alpha);

    }
}
