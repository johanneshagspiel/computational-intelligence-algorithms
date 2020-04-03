import java.io.File;
import java.io.IOException;
import java.util.*;

public class DataObject {

    /**
     * These arrays store the groups of feature vectors and their respective labels.
     */
    double[][] inputFeatures;
    double[][] validationFeatures, testFeatures, trainFeatures;
    int[][] validationLabels, testLabels, trainLabels;

    /**
     * These arrays store the training data divided into batches and their respective labels.
     */
    double[][][] batches;
    int[][][] batchlabels;


    public DataObject(String featureFile) throws IOException{
        int numfeatures = 10; //parsing becomes less efficient if we don't have access to these parameters,
        int numclasses = 7; //so we decided to add them in here for now. Some of our parsing code is also
        //not 100% flexible for other cases, but given we know the format, this is the best way.
        List<double[]> rawdata = new ArrayList<>();
        Scanner sc = new Scanner(new File(featureFile));
        Scanner sc1;
        while (sc.hasNextLine()) {
            sc1 = new Scanner(sc.nextLine());
            sc1.useDelimiter(","); //comma-separated lines, 1 line = 1 feature vector
            double[] features = new double[numfeatures]; //create fv
            int i = 0;
            while (sc1.hasNextDouble()) features[i++] = sc1.nextDouble(); //fill fv with given features

            rawdata.add(features); //add this object to list of objects
        }
        inputFeatures = new double[rawdata.size()][];

        for(int i = 0; i < rawdata.size(); i++){
            inputFeatures[i] = rawdata.get(i);
        }
    }
    /**
     * Instantiates a Data Object and divides the feature vectors and their respective labels into three groups.
     * @param featureFile      the path to the file from which to read the feature vectors
     * @param labelFile        the path to the file from which to read the labels belonging to said feature vectors
     * @param validatefraction the fraction of the data that should be reserved for validation
     * @param testfraction     the fraction of the data that should be put aside for testing
     * @param numperbatch      the number of items that should be in each batch
     * @throws IOException this will be thrown if something goes wrong while reading the files
     */
    public DataObject(String featureFile, String labelFile, double validatefraction, double testfraction, int numperbatch) throws IOException {
        int numfeatures = 10; //parsing becomes less efficient if we don't have access to these parameters,
        int numclasses = 7; //so we decided to add them in here for now. Some of our parsing code is also
                            //not 100% flexible for other cases, but given we know the format, this is the best way.

        List<double[]> rawdata = new ArrayList<>();
        Scanner sc = new Scanner(new File(featureFile));
        Scanner sc1;
        while (sc.hasNextLine()) {
            sc1 = new Scanner(sc.nextLine());
            sc1.useDelimiter(","); //comma-separated lines, 1 line = 1 feature vector
            double[] features = new double[numfeatures]; //create fv
            int i = 0;
            while (sc1.hasNextDouble()) features[i++] = sc1.nextDouble(); //fill fv with given features

            rawdata.add(features); //add this object to list of objects
        }

        List<int[]> labels = new ArrayList<>();
        sc = new Scanner(new File(labelFile));
        while (sc.hasNextInt()) {
            int[] label = new int[numclasses]; //we have a number of classes, we want it to classify as being class a and not being class b or c (etc.)
            label[sc.nextInt() - 1] = 1; //classes start at 1, arrays at 0. This saves a permanent 0 value in the output.
            labels.add(label); //add to list of labels
        }
        assert rawdata.size() == labels.size(); //should be the same size, otherwise they don't match

        Random r = new Random();
        double d;
        List<double[]> validate = new ArrayList<>(), test = new ArrayList<>(), train = new ArrayList<>();
        List<int[]> validateL = new ArrayList<>(), testL = new ArrayList<>(), trainL = new ArrayList<>();
        for (int i = 0; i < rawdata.size(); i++) {
            d = r.nextDouble(); //generates a value between 0 and 1, giving...
            if (d < validatefraction) { //a [validatefraction]% chance of being lower than validatefraction...
                validate.add(rawdata.get(i));
                validateL.add(labels.get(i));
            } else if (d - validatefraction < testfraction) { //a [testfraction]% chance of being between validatefraction and (testfraction + validatefraction)...
                test.add(rawdata.get(i));
                testL.add(labels.get(i));
            } else { //and a (100 - [validatefraction] - [testfraction])% chance of not being either of those, making this object belong to the train set
                train.add(rawdata.get(i));
                trainL.add(labels.get(i));
            }
        }

        //now, we convert the lists to arrays; we no longer need the extra, fancy methods lists provide
        validationFeatures = new double[validate.size()][];
        validationLabels = new int[validateL.size()][];
        for (int i = 0; i < validate.size(); i++) {
            validationFeatures[i] = validate.get(i);
            validationLabels[i] = validateL.get(i);
        }

        testFeatures = new double[test.size()][];
        testLabels = new int[testL.size()][];
        for (int i = 0; i < test.size(); i++) {
            testFeatures[i] = test.get(i);
            testLabels[i] = testL.get(i);
        }

        trainFeatures = new double[train.size()][];
        trainLabels = new int[trainL.size()][];
        for (int i = 0; i < train.size(); i++) {
            trainFeatures[i] = train.get(i);
            trainLabels[i] = trainL.get(i);
        }

        batchify(numperbatch);
    }

    /**
     * Redivides the objects in the test and train sets, possibly with a different division.
     * @param testfraction the percentage of objects to put in the test set rather than in the train set.
     * @param numperbatch  the number of items that should go in each batch
     */
    public void reshuffle(double testfraction, int numperbatch) {
        //once we have used the training and testing data, we can call this method to reshuffle the arrays
        //this way, we can use different input combinations to optimise our parameters, without touching the validation set
        List<double[]> splitdata = Arrays.asList(testFeatures);
        splitdata.addAll(Arrays.asList(trainFeatures));
        List<int[]> splitlabels = Arrays.asList(testLabels);
        splitlabels.addAll(Arrays.asList(trainLabels));
        Random r = new Random();
        double d;
        List<double[]> test = new ArrayList<>(), train = new ArrayList<>();
        List<int[]> testL = new ArrayList<>(), trainL = new ArrayList<>();
        for (int i = 0; i < splitdata.size(); i++) {
            d = r.nextDouble();
            if (d < testfraction) {
                test.add(splitdata.get(i));
                testL.add(splitlabels.get(i));
            } else {
                train.add(splitdata.get(i));
                trainL.add(splitlabels.get(i));
            }
        }

        testFeatures = new double[test.size()][];
        testLabels = new int[testL.size()][];
        for (int i = 0; i < test.size(); i++) {
            testFeatures[i] = test.get(i);
            testLabels[i] = testL.get(i);
        }

        trainFeatures = new double[train.size()][];
        trainLabels = new int[trainL.size()][];
        for (int i = 0; i < train.size(); i++) {
            trainFeatures[i] = train.get(i);
            trainLabels[i] = trainL.get(i);
        }

        batchify(numperbatch);
    }

    /**
     * Divide training data into batches.
     * @param numperbatch the number of feature vectors we put in each batch
     */
    public void batchify(int numperbatch) {
        int numbatches = (int) Math.ceil((1.0 * trainFeatures.length) / numperbatch);
        batches = new double[numbatches][][];
        batchlabels = new int[numbatches][][];
        for (int i = 0; i < numbatches; i++) {
            int size = Math.min(numperbatch, (trainFeatures.length - i*numperbatch));
            batches[i] = new double[size][];
            batchlabels[i] = new int[size][];
        }
        for (int i = 0; i < trainFeatures.length; i++) {
            batches[i/numperbatch][i%numperbatch] = trainFeatures[i];
            batchlabels[i/numperbatch][i%numperbatch] = trainLabels[i];
        }
    }
}
