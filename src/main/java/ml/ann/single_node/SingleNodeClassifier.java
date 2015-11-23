package ml.ann.single_node;

import ml.ann.NNUtils;
import ml.ann.weight.GivenWeightAssignment;
import ml.ann.weight.RandomWeightAssignment;
import ml.ann.weight.WeightAssignmentStrategy;
import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by Alvin Natawiguna on 11/23/2015.
 */
public abstract class SingleNodeClassifier extends AbstractClassifier {

    protected double learningRate;

    protected Double initialWeights;

    /**
     * The value of the bias
     */
    protected double bias = 1.0;

    public SingleNodeClassifier() {
        initialWeights = null;
        learningRate = 0.5;
    }

    public SingleNodeClassifier(double initialWeights) {
        this();
        this.initialWeights = initialWeights;
    }

    public SingleNodeClassifier(double initialWeights, double bias) {
        this(initialWeights);
        this.bias = bias;
    }

    public SingleNodeClassifier(double initialWeights, double bias, double learningRate) {
        this(initialWeights, bias);
        this.learningRate = learningRate;
    }

    public double getInitialWeights() {
        return initialWeights;
    }

    public void setInitialWeights(double initialWeights) {
        this.initialWeights = initialWeights;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        double inputVectors[][] = new double[data.size()][];
        for (int i = 0; i < data.size(); i++) {
            inputVectors[i] = NNUtils.instanceToInputVector(data.instance(i));
        }

        double targetVectors[][] = new double[data.size()][];
        for (int i = 0; i < data.size(); i++) {
            targetVectors[i] = NNUtils.instanceToTargetVector(data.instance(i));
        }

        this.classify(inputVectors, targetVectors);
    }

    protected abstract void classify(double[][] inputVectors, double[][] targetVectors);

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        return 0;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return new double[0];
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

        return result;
    }
}
