import java.io.File;
import java.io.IOException;
import java.util.*;

public class DataObject {

    double[][] validationFeatures, testFeatures, trainFeatures;
    int[][] validationLabels, testLabels, trainLabels;


    public DataObject(String FeatureFile, String labelFile, double validatefraction, double testfraction) throws IOException {
        int numfeatures = 10;
        int numclasses = 7;

        List<double[]> rawdata = new ArrayList<>();
        Scanner sc = new Scanner(new File(FeatureFile));
        Scanner sc1;
        while (sc.hasNextLine()) {
            sc1 = new Scanner(sc.nextLine());
            sc1.useDelimiter(",");
            double[] features = new double[numfeatures];
            int i = 0;
            while (sc1.hasNextDouble()) features[i++] = sc1.nextDouble();

            rawdata.add(features);
        }

        List<int[]> labels = new ArrayList<>();
        sc = new Scanner(new File(labelFile));
        while (sc.hasNextInt()) {
            int[] label = new int[numclasses];
            label[sc.nextInt() - 1] = 1;
            labels.add(label);
        }
        assert rawdata.size() == labels.size();

        Random r = new Random();
        double d;
        List<double[]> validate = new ArrayList<>(), test = new ArrayList<>(), train = new ArrayList<>();
        List<int[]> validateL = new ArrayList<>(), testL = new ArrayList<>(), trainL = new ArrayList<>();
        for (int i = 0; i < rawdata.size(); i++) {
            d = r.nextDouble();
            if (d < validatefraction) {
                validate.add(rawdata.get(i));
                validateL.add(labels.get(i));
            } else if (d - validatefraction < testfraction) {
                test.add(rawdata.get(i));
                testL.add(labels.get(i));
            } else {
                train.add(rawdata.get(i));
                trainL.add(labels.get(i));
            }
        }

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
    }

    public void reshuffle(double testfraction) {
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
    }
}
