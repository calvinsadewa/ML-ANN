package ml.ann.mlp;

import ml.ann.mlp.activation.ActivationStrategy;
import ml.ann.weight.WeightAssignmentStrategy;

/**
 * Created by calvin-pc on 11/19/2015.
 */
public class CustomActivationQuadraticCostLayer extends CustomActivationLayer {

    public CustomActivationQuadraticCostLayer(int num_input, int num_output, ActivationStrategy activation, WeightAssignmentStrategy ws) {
        super(num_input, num_output, activation,ws);
    }

    @Override
    public Double[] calculateOutputError(Double[] input, Double[] target) {
        // Using Quadratic Cost Function
        assert (input.length == numInput);
        assert (target.length == numOutput);
        Double[] output = feedfoward(input);
        Double[] delta = new Double[numOutput];
        Double[] deriv = this.derivActivate(this.weighted_input(input));
        for (int j = 0; j < output.length; j++) {
            delta[j] = (output[j] - target[j]) * deriv[j];
        }
        return delta;
    }
}
