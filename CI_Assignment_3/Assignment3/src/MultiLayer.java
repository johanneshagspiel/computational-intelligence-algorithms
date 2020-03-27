import ActivationFunctions.ActivationFunction;

import java.util.Arrays;

public class MultiLayer {

    //We use a lot of arrays of arrays, which are usually not 2d-arrays.
    //These arrays generally have the shape our MLP would have in a drawing, though sometimes
    //without the input (since that is not technically part of the MLP).
    //This is the most convenient format for storing perceptrons and values corresponding to them,
    //since it keeps the layered structure and semi-irregular (i.e. inputsize != hiddenlayersize != outputsizze) shape.

    Perceptron[][] neurons;
    int hiddenlayers;
    int neuronsperlayer;
    int inputs;
    int outputs;
    ActivationFunction activationFunction;

    public MultiLayer(int hl, int npl, int i, int o, ActivationFunction a) {
        hiddenlayers = hl;
        neuronsperlayer = npl;
        inputs = i;
        outputs = o;
        activationFunction = a;
        neurons = new Perceptron[hl + 1][]; //we need to store all hidden layers and the output layer. The input layer is given.
        neurons[0] = new Perceptron[npl];
        for (int j = 0; j < npl; j++) neurons[0][j] = new Perceptron(i, a); //all inputs feed into the first hidden layer, so this layer needs #inputs inputs for each perceptron
        for (int j = 1; j < hl; j++) {
            neurons[j] = new Perceptron[npl];
            for (int k = 0; k < npl; k++) neurons[j][k] = new Perceptron(npl, a); //all other hidden layers receive input from previous layer (i.e. have #neurons/layer inputs)
        }
        neurons[hl] = new Perceptron[o];
        for (int j = 0; j < o; j++) neurons[hl][j] = new Perceptron(npl, a); //output neurons receive from last hidden layer (i.e. also need #neurons/layer inputs)
    }

    public double[][] process(double[] input) {
        double[][] res = new double[hiddenlayers + 2][]; //this will represent all values perceptrons take on, including input "neurons"
        res[0] = input;
        for (int i = 0; i < hiddenlayers; i++) {
            res[i+1] = new double[neuronsperlayer];
            for (int j = 0; j < neuronsperlayer; j++) {
                res[i+1][j] = neurons[i][j].activation(res[i]); //for each hidden layer, calculate new result from prev result
            }
        }
        res[hiddenlayers+1] = new double[outputs];
        for (int i = 0; i < outputs; i++) {
            res[hiddenlayers+1][i] = neurons[hiddenlayers][i].activation(res[hiddenlayers]); //use output of last hidden layer to find output array
        }
        return res;
    }

    public void backPropagate(double[][] results, int[] labels, double alpha) {
        assert results.length == hiddenlayers + 2;
        assert results[hiddenlayers+1].length == outputs;
        assert results[0].length == inputs;
        for (int i = 1; i < hiddenlayers+1; i++) assert results[i].length == neuronsperlayer;
        assert labels.length == outputs;

        double[][] prevs = new double[hiddenlayers + 1][]; //store all chained values
        prevs[hiddenlayers] = new double[outputs];
        for (int i = 0; i < outputs; i++) {
            Perceptron current = neurons[hiddenlayers][i];
            prevs[hiddenlayers][i] = (labels[i] - ((int) (results[hiddenlayers+1][i] + 0.5))) * activationFunction.getDerivative(current.preprocess(results[hiddenlayers]), current.threshold);
        }
        for (int i = hiddenlayers - 1; i >= 0; i--) {
            prevs[i] = new double[neuronsperlayer];
            for (int j = 0; j < neuronsperlayer; j++) {
                prevs[i][j] = 0;
                Perceptron current = neurons[i][j];
                for (int k = 0; k < prevs[i+1].length; k++) prevs[i][j] += prevs[i+1][k] * neurons[i+1][k].weightArray[j] * activationFunction.getDerivative(current.preprocess(results[i]), current.threshold);
            }
        }
        for (int i = 0; i <= hiddenlayers; i++) { //all layers, including ouput, have to have their weights updated
            for (int j = 0; j < neurons[i].length; j++) {
                for (int k = 0; k < neurons[i][j].weightArray.length; k++) neurons[i][j].weightArray[k] += alpha * prevs[i][j] * results[i][k];
                neurons[i][j].threshold -= alpha * prevs[i][j];
            }
        }
    }

    public void run(int epoch, double alpha, double[][] inputArray, int[][] desiredResultArray) {
        double firsterror = -1;
        for (int i = 0; i < epoch; i++) {
            double epocherror = 0;
            for (int j = 0; j < inputArray.length; j++) {
                double[][] res = process(inputArray[j]);
                double totalerror = 0;
//                System.out.println(Arrays.toString(res[res.length-1]));
//                System.out.println(Arrays.toString(desiredResultArray[j]));
                for (int k = 0; k < res[res.length - 1].length; k++) totalerror += Math.abs(((int) (res[res.length - 1][k] + 0.5)) - desiredResultArray[j][k]);
                System.out.println("Average error of iteration " + j + ": " + totalerror/res[res.length - 1].length);
                epocherror += totalerror/res[res.length - 1].length;
                backPropagate(res, desiredResultArray[j], alpha);
            }
            System.out.println("Total error of epoch " + i + ": " + epocherror/inputArray.length);
            if (i == 0) firsterror = epocherror/inputArray.length;
        }
        System.out.println("We started at " + firsterror);
    }
}
