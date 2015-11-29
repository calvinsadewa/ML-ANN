package ml.ann.single_node;

import weka.core.Utils;
import weka.core.logging.Logger;

/**
 * Created by Alvin Natawiguna on 11/23/2015.
 */
public class DeltaRuleIncremental extends SingleNodeClassifier {

    int numClasses;

    public DeltaRuleIncremental(int numClasses) {
        super();
        this.numClasses = numClasses;
    }

    public DeltaRuleIncremental(int numClasses, double learningRate, Double initialWeight,
                      int maxEpoch, double minError,
                      double bias, boolean debug)
    {
        super(learningRate, initialWeight, maxEpoch, minError, bias, debug);
        this.numClasses = numClasses;
    }

    @Override
    public void buildClassifier(double[][] inputVectors, double[] targetVector) {
        int classCount = inputVectors[0].length;
        int instanceCount = inputVectors.length;

        initializeWeights(classCount);

        int currentEpoch = 0;
        double currentError = Double.MAX_VALUE;

        while (currentEpoch < maxEpoch && Utils.gr(currentError, minError)) {
            for (int i = 0; i < instanceCount; i++) {
                // 1. calculate initial output
                double output = classify(inputVectors[i]);

                // 2. calculate error
                double error = targetVector[i] - output;

                // 3. update the weights
                // start from bias
                weights[0] += learningRate * bias * error;
                for (int j = 0; j < classCount; j++) {
                    weights[j+1] += learningRate * inputVectors[i][j] * error;
                }
            }

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
        assert(weights.length == inputVector.length + 1);

        // calculate the bias first
        double outputValue = bias * weights[0];
        for (int i = 0; i < inputVector.length; i++) {
            outputValue += inputVector[i] * weights[i+1];
        }

        return outputValue;
    }
}
