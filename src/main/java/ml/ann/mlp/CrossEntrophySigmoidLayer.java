package ml.ann.mlp;

import ml.ann.mlp.activation.SigmoidActivation;
import ml.ann.weight.WeightAssignmentStrategy;

/**
 * Created by calvin-pc on 11/19/2015.
 */
public class CrossEntrophySigmoidLayer extends CustomActivationLayer {
    public CrossEntrophySigmoidLayer(int num_input, int num_output, WeightAssignmentStrategy ws) {
        super(num_input, num_output, new SigmoidActivation(), ws);
    }

    @Override
    // Calculate error in output layer if it's last layer
    public Double[] calculateOutputError(Double[] input, Double[] target) {
        // Using Cross Enthropy Function
        assert (input.length == numInput);
        assert (target.length == numOutput);
        Double[] output = feedfoward(input);
        Double[] delta = new Double[numOutput];
        for (int j = 0; j < output.length; j++) {
            delta[j] = (output[j] - target[j]);
        }
        return delta;
    }
}
