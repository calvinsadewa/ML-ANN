package ml.ann.single_node;

import ml.ann.weight.GivenWeightAssignment;
import ml.ann.weight.RandomWeightAssignment;
import ml.ann.weight.WeightAssignmentStrategy;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Alvin Natawiguna on 11/29/2015.
 */
public abstract class SingleNodeClassifier {

    protected double learningRate = 0.1;

    protected Double initialWeight = null;

    protected int maxEpoch = 1000;

    protected double minError = 0;

    protected double bias = 1.0;

    protected boolean debug = true;

    protected double[] weights;

    public SingleNodeClassifier() {

    }

    public SingleNodeClassifier(double learningRate, Double initialWeight, int maxEpoch, double minError, double bias, boolean debug) {
        this.learningRate = learningRate;
        this.initialWeight = initialWeight;
        this.maxEpoch = maxEpoch;
        this.minError = minError;
        this.bias = bias;
        this.debug = debug;
    }

    public abstract void buildClassifier(double[][] inputVectors, double[] targetVector);

    /**
     * Classifies the current instance with the current model.
     * @param inputVector a vector of attributes
     * @return the number that represents the class's value
     */
    public abstract double classify(double[] inputVector);

    public abstract double[] distribution(double[] inputVector);

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

    public double[] getWeights() {
        return weights;
    }

    /**
     * Initializes the weights of the neural network
     * Fills the first 'row' of weights with the initial weight number
     *
     * @param attributeCount the number of attributes
     */
    protected void initializeWeights(int attributeCount) {
        WeightAssignmentStrategy strategy;
        if (initialWeight == null) {
            strategy = new RandomWeightAssignment();
        } else {
            strategy = new GivenWeightAssignment(initialWeight);
        }

        // initialize all of the class's weights
        // the zeroth element is the bias
        weights = new double[attributeCount + 1];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = strategy.getWeight();
        }
    }

    protected double calculateErrorSum(double[][] inputVectors, double[] targetVector) {
        // 4. calculate final output
        double errorSum = 0;
        for (int i = 0; i < inputVectors.length; i++) {
            double output = classify(inputVectors[i]);

            // calculate error
            double error = targetVector[i] - output;

            errorSum += Math.pow(error, 2);
        }

        return errorSum;
    }
}
