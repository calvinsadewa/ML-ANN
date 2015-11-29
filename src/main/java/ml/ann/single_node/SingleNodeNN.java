package ml.ann.single_node;

import ml.ann.util.DeepCopy;
import ml.ann.util.NNUtils;
import weka.classifiers.AbstractClassifier;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.util.ArrayList;

/**
 * Created by Alvin Natawiguna on 11/29/2015.
 */
public class SingleNodeNN extends AbstractClassifier {

    protected double learningRate = 0.1;

    protected Double initialWeight = null;

    protected int maxEpoch = 1000;

    protected double minError = 0;

    protected double bias = 1.0;

    protected String classifierName = "perceptron";

    protected SingleNodeClassifier classifier;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        // preprocessing: data filtering and cleaning
        Instances trainingData = (Instances) DeepCopy.copy(data);
        trainingData.deleteWithMissingClass();

        // convert missing values
        Filter missingValuesFilter = new ReplaceMissingValues();
        missingValuesFilter.setInputFormat(trainingData);
        trainingData = Filter.useFilter(trainingData, missingValuesFilter);

        // convert to input vectors
        double inputVectors[][] = new double[trainingData.numInstances()][];
        for (int i = 0; i < trainingData.numInstances(); i++) {
            inputVectors[i] = NNUtils.instanceToInputVector(trainingData.instance(i));
        }

        double targetVector[];
        // initialize the classifier
        switch (classifierName) {
            case "perceptron":
                classifier = new Perceptron(learningRate, initialWeight, maxEpoch, minError, bias, getDebug());
                targetVector = NNUtils.instancesToTargetVector(trainingData, true);
                break;
            case "delta-inc":
                classifier = new DeltaRuleIncremental(trainingData.numClasses(), learningRate,
                                initialWeight, maxEpoch, minError, bias, getDebug());
                targetVector = NNUtils.instancesToTargetVector(trainingData);
                break;
            case "delta-batch":
                classifier = new DeltaRuleBatch(trainingData.numClasses(), learningRate,
                        initialWeight, maxEpoch, minError, bias, getDebug());
                targetVector = NNUtils.instancesToTargetVector(trainingData);
                break;
            default:
                throw new Exception("Unknown classifier: " + classifierName);
        }

        classifier.buildClassifier(inputVectors, targetVector);
    }

    @Override
    public double classifyInstance(Instance instance) {
        assert(classifier != null);

        return classifier.classify(NNUtils.instanceToInputVector(instance));
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        // values
        result.enable(Capabilities.Capability.MISSING_VALUES);

        return result;
    }

    @Override
    public String [] getOptions() {
        ArrayList<String> options = new ArrayList<>();
        if (getDebug()) {
            options.add("-D");
        }

        options.add("-U " + learningRate);
        options.add("-Bias " + bias);
        options.add("-Epochs " + maxEpoch);
        options.add("-Error " + minError);
        options.add("-Classifier " + classifierName);

        if (initialWeight != null) {
            options.add("-Weight " + initialWeight);
        }

        return (String[]) options.toArray();
    }

    /**
     * Parses a given list of options. Valid options are:
     *
     * <p>
     * -D  <br>
     * If set, classifierName is run in debug mode and
     * may output additional info to the console.
     * </p>
     *
     * <p>
     * -Bias [double] <br>
     * If set, classifierName will set the bias of the node. Defaults to 1.0.
     * </p>
     *
     * <p>
     * -U [double] <br>
     * If set, the classifierName will be run with the defined learning rate. Defaults to 0.1.
     * </p>
     *
     * <p>
     * -Epoch [int] <br>
     * If set, the classifierName will be run until the defined number of epochs is reached. Defaults to 1000.
     * </p>
     *
     * <p>
     * -Weight [double] <br>
     * If set, classifierName will set the initial weights of the node. Defaults to random.
     * </p>
     *
     * <p>
     * -Error [double] <br>
     * If set, classifierName will set the minimum error that defines the classifierName as 'converged'. Defaults to 0.
     * </p>
     *
     * <p>
     * -Classifier (perceptron|delta-inc|delta-batch) <br>
     * Sets the classifierName to be used. Defaults to perceptron.
     * </p>
     *
     * @param options the list of options as an array of strings
     * @exception Exception if an option is not supported
     */
    @Override
    public void setOptions(String[] options) throws Exception {
        super.setOptions(options);

        String bias = Utils.getOption("Bias", options);
        setBias(bias.isEmpty() ? 1.0 : Double.parseDouble(bias));

        String learningRate = Utils.getOption('U', options);
        setLearningRate(learningRate.isEmpty() ? 0.1 : Double.parseDouble(learningRate));

        String epoch = Utils.getOption("Epoch", options);
        setMaxEpoch(epoch.isEmpty() ? 1000 : Integer.parseInt(epoch));

        String initialWeight = Utils.getOption("Weight", options);
        setInitialWeight(initialWeight.isEmpty() ? null : Double.parseDouble(initialWeight));

        String classifier = Utils.getOption("Classifier", options).toLowerCase();
        switch (classifier) {
            case "perceptron":
            case "delta-inc":
            case "delta-batch":
                this.classifierName = classifier;
                break;
            default:
                throw new Exception("Unknown classifier: " + classifier);
        }
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public Double getInitialWeight() {
        return initialWeight;
    }

    public void setInitialWeight(Double initialWeight) {
        this.initialWeight = initialWeight;
    }

    public int getMaxEpoch() {
        return maxEpoch;
    }

    public void setMaxEpoch(int maxEpoch) {
        this.maxEpoch = maxEpoch;
    }

    public double getMinError() {
        return minError;
    }

    public void setMinError(double minError) {
        this.minError = minError;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }
}
