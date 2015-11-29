package ml.ann.single_node;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.logging.Logger;

import java.util.List;

/**
 * Created by Alvin Natawiguna on 11/23/2015.
 */
public class DeltaRuleBatch extends SingleNodeClassifier {

    protected int numClasses;

    public DeltaRuleBatch(int numClasses) {
        super();
        this.numClasses = numClasses;
    }

    public DeltaRuleBatch(int numClasses, double learningRate, Double initialWeight,
                          int maxEpoch, double minError,
                          double bias, boolean debug) {
        super(learningRate, initialWeight, maxEpoch, minError, bias, debug);
        this.numClasses = numClasses;
    }

    @Override
    public void buildClassifier(double[][] inputVectors, double[] targetVector) {
        int classCount = inputVectors[0].length;
        int instanceCount = inputVectors.length;
        int attributeCount = inputVectors[0].length;

        initializeWeights(classCount);

        int currentEpoch = 0;
        double currentError = Double.MAX_VALUE;

        while (currentEpoch < maxEpoch && Utils.gr(currentError, minError)) {
            double errorVectors[][] = new double[instanceCount][];

            for (int i = 0; i < instanceCount; i++) {
                // 1. calculate initial output
                double output = classify(inputVectors[i]);

                // 2. calculate error
                double error = targetVector[i] - output;

                // 3. calculate the error from the instance
                double errorVector[] = new double[attributeCount + 1];

                // starting with the bias
                errorVector[0] = error * bias;
                for (int j = 1; j < attributeCount + 1; j++) {
                    errorVector[j] = error * inputVectors[i][j - 1];
                }

                // 4. save it for later use
                errorVectors[i] = errorVector;
            }

            // 5. calculate the weight updates
            assert (weights.length == errorVectors[0].length);
            for (int i = 0; i < weights.length; i++) {
                double sumError = 0;

                for (int j = 0; j < instanceCount; j++) {
                    sumError += errorVectors[j][i];
                }

                weights[i] += sumError * learningRate;
            }

            // 6. calculate error
            currentError = calculateErrorSum(inputVectors, targetVector);

            // divide by number of classes defined
            currentError /= numClasses;
            currentEpoch++;

            if (debug) {
                Logger.log(Logger.Level.FINE, String.format("Epoch: %d, MSE: %f", currentEpoch, currentError));
            }
        }
    }

    @Override
    public double classify(double[] inputVector) {
        // calculate the bias first
        double outputValue = bias * weights[0];
        for (int i = 1; i < weights.length; i++) {
            outputValue += inputVector[i-1] * weights[i];
        }

        return outputValue;
    }

    @Override
    public double[] distribution(double[] inputVector) {
        double dist[] = new double[numClasses];
        double output = classify(inputVector);

        if (numClasses == 1) {
            dist[0] = output;
        } else {
            dist[(int)Math.round(output)] = 1;
        }

        return dist;
    }
}
