package ml.ann.single_node;

import weka.core.Utils;
import weka.core.logging.Logger;
import weka.core.logging.Logger.Level;

/**
 * Created by Alvin Natawiguna on 11/23/2015.
 */
public class Perceptron extends SingleNodeClassifier {
    public Perceptron() {
        super();
    }

    public Perceptron(double learningRate, Double initialWeight,
                      int maxEpoch, double minError,
                      double bias, boolean debug)
    {
        super(learningRate, initialWeight, maxEpoch, minError, bias, debug);
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

            // 4. calculate final output
            currentError = calculateErrorSum(inputVectors, targetVector);

            // divide by two, since we have binary classes
            currentError *= 0.5;
            currentEpoch++;

            if (debug) {
                Logger.log(Level.FINE, String.format("Epoch: %d, MSE: %f", currentEpoch, currentError));
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

        // apply sign function
        // we use 0 and 1 here
        return outputValue > 0 ? 0 : 1;
    }
}
