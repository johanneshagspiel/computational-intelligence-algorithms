import ActivationFunctions.ActivationFunction;

import java.io.FileWriter;
import java.io.IOException;
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

    double[][][] currentBatchGradients;
    double[][][] prevBatchGradients;

    /**
     * Constructor for our multilayer perceptron. Creates the packaging object and
     * fills the neurons array with perceptrons.
     *
     * @param hl  the number of hidden layers our MLP will have
     * @param npl the number of perceptrons we will put in each hidden layer
     * @param i   the number of input features we should expect
     * @param o   the number of outputs we should give
     * @param a   the activation function our perceptrons should use
     */
    public MultiLayer(int hl, int npl, int i, int o, ActivationFunction a) {
        hiddenlayers = hl;
        neuronsperlayer = npl;
        inputs = i;
        outputs = o;
        activationFunction = a;
        neurons = new Perceptron[hl + 1][]; //we need to store all hidden layers and the output layer. The input layer is given.
        neurons[0] = new Perceptron[npl];
        for (int j = 0; j < npl; j++)
            neurons[0][j] = new Perceptron(i, a); //all inputs feed into the first hidden layer, so this layer needs #inputs inputs for each perceptron
        for (int j = 1; j < hl; j++) {
            neurons[j] = new Perceptron[npl];
            for (int k = 0; k < npl; k++)
                neurons[j][k] = new Perceptron(npl, a); //all other hidden layers receive input from previous layer (i.e. have #neurons/layer inputs)
        }
        neurons[hl] = new Perceptron[o];
        for (int j = 0; j < o; j++)
            neurons[hl][j] = new Perceptron(npl, a); //output neurons receive from last hidden layer (i.e. also need #neurons/layer inputs)

        initCurBatchGrads();
        prevBatchGradients = currentBatchGradients; //trust me, I'm an engineer **
    }

    /**
     * Method used to set the gradients of the current batch to 0 to be able to redetermine them.
     */
    public void initCurBatchGrads() {
        currentBatchGradients = new double[neurons.length][][]; //for each layer...
        for (int i = 0; i < neurons.length; i++) {
            currentBatchGradients[i] = new double[neurons[i].length][]; //...we give each neuron...
            for (int j = 0; j < neurons[i].length; j++)
                //...an array for all weights, including the threshold
                currentBatchGradients[i][j] = new double[neurons[i][j].weightArray.length + 1];
        }
    }

    /**
     * Runs an input vector through the network and returns the output and all intermediate results.
     *
     * @param input the input vector we should work on
     * @return the output vector we got by doing this
     */
    public double[][] process(double[] input) {
        double[][] res = new double[hiddenlayers + 2][]; //this will represent all values perceptrons take on, including input "neurons"
        res[0] = input;
        for (int i = 0; i < hiddenlayers; i++) {
            res[i + 1] = new double[neuronsperlayer];
            for (int j = 0; j < neuronsperlayer; j++) {
                res[i + 1][j] = neurons[i][j].activation(res[i]); //for each hidden layer, calculate new result from prev result
            }
        }
        res[hiddenlayers + 1] = new double[outputs];
        for (int i = 0; i < outputs; i++) {
            res[hiddenlayers + 1][i] = neurons[hiddenlayers][i].activation(res[hiddenlayers]); //use output of last hidden layer to find output array
        }
        return res;
    }

    /**
     * Uses the result of processing a vector to update all weights.
     *
     * @param results the result of calling process with an appropriate input vector
     * @param labels  the labels that belong to the objects portrayed by said input vector
     */
    public void backPropagate(double[][] results, int[] labels) {
        assert results.length == hiddenlayers + 2;
        assert results[hiddenlayers + 1].length == outputs;
        assert results[0].length == inputs;
        for (int i = 1; i < hiddenlayers + 1; i++) assert results[i].length == neuronsperlayer;
        assert labels.length == outputs;

        double[][] prevs = new double[hiddenlayers + 1][]; //store all chained values
        prevs[hiddenlayers] = new double[outputs];
        for (int i = 0; i < outputs; i++) {
            Perceptron current = neurons[hiddenlayers][i];
            //the change in error over the first weights is equal to the error (first term)
            //multiplied by the derivative of the activation function (second term)
            //multiplied by the previous result (not shown, cannot be reused)
            prevs[hiddenlayers][i] = (labels[i] - ((int) (results[hiddenlayers + 1][i] + 0.5))) //we round our result to either 1 or 0; if it already classifies class a as a, we don't need to change those weights
                    * activationFunction.getDerivative(current.preprocess(results[hiddenlayers]), current.threshold);
        }
        for (int i = hiddenlayers - 1; i >= 0; i--) {
            prevs[i] = new double[neuronsperlayer];
            for (int j = 0; j < neuronsperlayer; j++) {
                prevs[i][j] = 0;
                Perceptron current = neurons[i][j];
                for (int k = 0; k < prevs[i + 1].length; k++) //sum over all outputs
                    prevs[i][j] += prevs[i + 1][k] //part we alredy had
                            * neurons[i + 1][k].weightArray[j] //chain rule to get to our current perceptron ( * weight * derivative of activation)
                            * activationFunction.getDerivative(current.preprocess(results[i]), current.threshold);
            }
        }
        for (int i = 0; i <= hiddenlayers; i++) { //all layers, including ouput, have to have their weights (and threshold updated)
            for (int j = 0; j < neurons[i].length; j++) {
                for (int k = 0; k < neurons[i][j].weightArray.length; k++)
                    currentBatchGradients[i][j][k] += prevs[i][j] * results[i][k];
                currentBatchGradients[i][j][neurons[i][j].weightArray.length] += prevs[i][j];
            }
        }
    }

    /**
     * Perform the actual updates on the weights and thresholds.
     *
     * @param alpha the learning rate
     * @param beta  the momentum factor
     */
    public void doGradientDescent(double alpha, double beta) {
        for (int i = 0; i <= hiddenlayers; i++) { //all layers, including ouput, have to have their weights (and threshold) updated
            for (int j = 0; j < neurons[i].length; j++) {
                Perceptron current = neurons[i][j];
                for (int k = 0; k < current.weightArray.length; k++)
                    current.weightArray[k] += alpha * (prevBatchGradients[i][j][k] = beta * prevBatchGradients[i][j][k] + (1 - beta) * currentBatchGradients[i][j][k]);
                current.threshold -= alpha * (prevBatchGradients[i][j][current.weightArray.length] = beta * prevBatchGradients[i][j][current.weightArray.length] + (1 - beta) * currentBatchGradients[i][j][current.weightArray.length]);
                //threshold has inverse effect, so -= i.o. +=
            }
        }
    }

    /**
     * Shortcut method to train the MLP on an array of inputs and outputs.
     *
     * @param epoch       the number of times we should train our network with all given objects
     * @param alpha       the learning rate we use to update the weights
     * @param batches     the array of batches, with each batch being an array of feature vectors we should train on
     * @param batchLabels the array of labels belonging to each batch
     * @param beta        the momentum factor
     * @param heuristics  turns the usage of heuristics on or off
     *
     * @return            returns the training error
     */
    public double run(int epoch, double alpha, double[][][] batches, int[][][] batchLabels, double beta, boolean heuristics) {
        assert batches.length == batchLabels.length;
        double firsterror = -1; //store the error of the first round, mainly so we can see how well it trained
        double prevprevError = -1;
        double prevError = -1;

        for (int i = 0; i < epoch; i++) {
            double epochErr = 0;
            for (int j = 0; j < batches.length; j++) {
                double batchError = 0;
                initCurBatchGrads(); //besides it just generally being better to have this at the start of the current batch,
                //it also makes sure it no longer shares a reference with prevBatchGradients (** told you to trust me)
                double[][] inputArray = batches[j];
                int[][] desiredResultArray = batchLabels[j];
                assert inputArray.length == desiredResultArray.length;
                for (int k = 0; k < inputArray.length; k++) {
                    double[][] res = process(inputArray[k]);
                    double totalerror = 0;
                    for (int l = 0; l < res[res.length - 1].length; l++) //for each class, check if our MLP says this object is in it
                        totalerror += Math.abs(((int) (res[res.length - 1][l] + 0.5)) - desiredResultArray[k][l]); //does it say it wrong? Error++
                    //System.out.println("Average error of iteration " + k + ": " + totalerror / res[res.length - 1].length); //show relative error
                    batchError += totalerror / res[res.length - 1].length; //add this error to the total error made in this epoch
                    backPropagate(res, desiredResultArray[k]);
                }
                double avgBatchErr = batchError / inputArray.length;
                //System.out.println("Total average error of batch " + j + ": " + avgBatchErr); //show the average error it made on objects in this batch
                doGradientDescent(alpha, beta);
                epochErr += avgBatchErr;
            }
            double avgEpochErr = epochErr / batches.length;
            double epsilon = 0.00025;
            if (i == 0) firsterror = avgEpochErr;
            //System.out.println("Total average error op epoch " + i + ": " + avgEpochErr);
            if (Math.abs(prevError - avgEpochErr) <= epsilon && Math.abs(prevprevError - prevError) <= epsilon && Math.abs(prevprevError - avgEpochErr) <= epsilon) {
                prevprevError = prevError;
                prevError = avgEpochErr;
                break;
            }
                //if there's barely been any change in the past 2 rounds, we're probably done
                //we chose to use error rather than gradient since we already have access to it,
                //and because it is about as good a measure as the gradient itself
                //(a small change in error usually means little to no change in the parameters)

            if(heuristics)
            {
                if((avgEpochErr*1.04) > prevError)
                    alpha *= 0.7;
                if((prevError*1.04) > avgEpochErr)
                    alpha *= 1.05;
            }

            prevprevError = prevError;
            prevError = avgEpochErr;
        }
        //System.out.println("We started at " + firsterror + " and ended at " + prevError);
        return prevError;
    }

    /**
     * Shortcut method to test the MLP on an array of inputs and outputs.
     *
     * @param input  the array of feature vectors we should test our MLP on
     * @param labels the labels that should be given after processing the respective inputs
     *
     * @return      returns the testing error
     */
    public double test(double[][] input, int[][] labels) {
        assert input.length == labels.length;
        double[][] res;
        int errors = 0;
        for (int i = 0; i < input.length; i++) { //process all inputs and see how well we can predict classes
            //difference with error calculation in run is that we now reduce the classification from yes/no for each class to one class;
            //the one with the highest probability
            res = process(input[i]);
            assert res[res.length - 1].length == labels[i].length;
            int maxidx = 0;
            for (int j = 1; j < res[res.length - 1].length; j++) { //start at 1 since 0 is default
                if (res[res.length - 1][j] > res[res.length - 1][maxidx])
                    maxidx = j; //higher probability? This is now predicted class
            }
            if (labels[i][maxidx] != 1) errors++; //wrong? Error++
        }
        //System.out.println("Testing error " + (1.0 * errors) / labels.length); //make error double i.o. int and divide by number of classified cases for relative error
        return  (1.0 * errors) / labels.length;
    }

    /**
     * Shortcut method to create a confusion matrix.
     *
     * @param input  the array of feature vectors we should test our MLP on
     * @param labels the labels that should be given after processing the respective inputs
     *
     * @return      returns the confusion matrix
     */
    public int[][] confusionMatrix(double[][] input, int[][] labels) {

        assert input.length == labels.length;
        double[][] res;
        int[][] confusionMatrix = new int[7][7]; //we know there are only 7 classes
        int[] result = new int[input.length];

        for (int i = 0; i < input.length; i++) { //process all inputs and see how well we can predict classes
            //difference with error calculation in run is that we now reduce the classification from yes/no for each class to one class;
            //the one with the highest probability
            res = process(input[i]);
            assert res[res.length - 1].length == labels[i].length;
            int maxidx = 0;
            for (int j = 1; j < res[res.length - 1].length; j++) { //start at 1 since 0 is default
                if (res[res.length - 1][j] > res[res.length - 1][maxidx])
                    maxidx = j; //higher probability? This is now predicted class
            }
            result[i] = maxidx;
        }

        for (int i = 0; i < result.length; i++) {
                int prediction = result[i];
                int actual = 0;

                for (int j = 0; j < labels[i].length; j++) {
                    if (labels[i][j] == 1) actual = j; //get which class this instance actually belongs to
                }

                confusionMatrix[actual][prediction] ++; //increase the entry in the confusion matrix
            }
        return confusionMatrix;
    }


    /**
     * Shortcut method to create a csv file with only the predicted classes.
     *
     * @param input  the feature vectors which we want to determine the associated classes of
     */
    public void output(double[][] input) throws IOException {
        double[][] res;
        int[] outputs = new int[input.length];
        for (int i = 0; i < input.length; i++) { //process all inputs and see how well we can predict classes
            //difference with error calculation in run is that we now reduce the classification from yes/no for each class to one class;
            //the one with the highest probability
            res = process(input[i]);
            int maxidx = 0;
//            System.out.println(Arrays.toString(res[res.length-1]));
//            System.out.println(res[res.length-1].length);
//            System.out.println(res.length);
            for (int j = 0; j < res[res.length - 1].length; j++) { //start at 1 since 0 is default
                if (res[res.length - 1][j] > res[res.length - 1][maxidx])
                    maxidx = j; //higher probability? This is now predicted class
            }
            outputs[i] = maxidx + 1;
        }
        FileWriter writer = new FileWriter("Group_8_classes.txt");
        for(int i : outputs) {
            writer.write(i + ",");
        }
        writer.close();
    }


    /**
     * Shortcut method to return the number of epochs necessary to train a mlp until it reaches a certain convergence criteria
     *
     * @param beta        the momentum factor
     * @param alpha       the learning rate we use to update the weights
     * @param batches     the array of batches, with each batch being an array of feature vectors we should train on
     * @param batchLabels the array of labels belonging to each batch
     * @param convergence the convergence criteria
     * @param heuristics  option to turn the usage of heuristics on or off
     *
     * @return            the number of epochs needed to train the mlp to reach a certain convergence criteria
     *
     */
    public int getEpochsToConvergence(double beta, double alpha, double[][][] batches, int[][][] batchLabels, double convergence, boolean heuristics) {

        assert batches.length == batchLabels.length;
        double firsterror = Double.MAX_VALUE; //store the error of the first round, mainly so we can see how well it trained
        double prevprevError = -1;
        double prevError = -1;

        int epoch = 0;

        while(firsterror > convergence) {

            double epochErr = 0;
            for (int j = 0; j < batches.length; j++) {
                double batchError = 0;
                initCurBatchGrads(); //besides it just generally being better to have this at the start of the current batch,
                //it also makes sure it no longer shares a reference with prevBatchGradients (** told you to trust me)
                double[][] inputArray = batches[j];
                int[][] desiredResultArray = batchLabels[j];
                assert inputArray.length == desiredResultArray.length;
                for (int k = 0; k < inputArray.length; k++) {
                    double[][] res = process(inputArray[k]);
                    double totalerror = 0;
                    for (int l = 0; l < res[res.length - 1].length; l++) //for each class, check if our MLP says this object is in it
                        totalerror += Math.abs(((int) (res[res.length - 1][l] + 0.5)) - desiredResultArray[k][l]); //does it say it wrong? Error++
                    batchError += totalerror / res[res.length - 1].length; //add this error to the total error made in this epoch
                    backPropagate(res, desiredResultArray[k]);
                }
                double avgBatchErr = batchError / inputArray.length;
                doGradientDescent(alpha, beta);
                epochErr += avgBatchErr;
            }
            double avgEpochErr = epochErr / batches.length;
            double epsilon = 0.00025;
            if (epoch == 0) firsterror = avgEpochErr;
            if (Math.abs(prevError - avgEpochErr) <= epsilon && Math.abs(prevprevError - prevError) <= epsilon && Math.abs(prevprevError - avgEpochErr) <= epsilon)
                //if there's barely been any change in the past 2 rounds, we're probably done
                //we chose to use error rather than gradient since we already have access to it,
                //and because it is about as good a measure as the gradient itself
                //(a small change in error usually means little to no change in the parameters)
                break;

            if(heuristics)
            {
                if((avgEpochErr*1.04) > prevError)
                    alpha *= 0.7;
                if((prevError*1.04) > avgEpochErr)
                    alpha *= 1.05;
            }

            prevprevError = prevError;
            prevError = avgEpochErr;
            epoch++;
        }
        return epoch;
    }
}

