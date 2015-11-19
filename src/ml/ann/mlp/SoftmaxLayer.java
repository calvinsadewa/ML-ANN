package ml.ann.mlp;

import ml.ann.mlp.weight.WeightAssignmentStrategy;

/**
 * Created by calvin-pc on 11/19/2015.
 */
public class SoftmaxLayer implements RegularizedMomentumLayer {

    // Weight matrix
    // use like "weights[no_neuron][input]"
    public Double[][] weights;
    public Double[][] momentum;
    public Double[] bias;

    public double momentum_rate = 0;
    public int numInput;
    public int numOutput;

    public void setMomentum(Double momentum) {momentum_rate = momentum;}
    public double getMomentum() {return momentum_rate;}

    public SoftmaxLayer(int num_input, int num_output, WeightAssignmentStrategy ws) {
        this.numInput = num_input;
        this.numOutput = num_output;
        weights = new Double[num_output][];
        momentum = new Double[num_output][];
        bias = new Double[num_output];
        //Randomize the weight matrix
        for (int i = 0; i < num_output; i++) {
            weights[i] = new Double[num_input];
            momentum[i] = new Double[num_input];
            for (int j = 0; j < num_input; j++) {
                weights[i][j] = ws.getWeight();
                momentum[i][j] = 0.0;
            }
            bias[i] = ws.getWeight();
        }
    }

    public int getNumInput() {
        return numInput;
    }

    public int getNumOutput() {
        return numOutput;
    }

    // return the array of neuron_weight dot. input + bias
    public Double[] weighted_input(Double[] input) {
        assert(input.length == numInput);
        Double[] summation = new Double[numOutput];
        for (int i = 0; i < numOutput; i++) {
            Double[] neuron_vec = weights[i];

            Double sum = 0.0;
            // do input dot neuron_vec
            for (int j = 0; j < numInput; j++) {
                sum = sum + input[j] * neuron_vec[j];
            }

            //add bias
            summation[i] = sum + bias[i];
        }
        return summation;
    }

    public Double[] activate(Double[] input) {
        assert(input.length == numInput);
        Double[] ret = new Double[input.length];
        Double sum = 0.0;
        for (int i = 0; i < input.length; i++) {
            ret[i] = Math.exp(input[i]);
            sum = sum + ret[i];
        }
        for (int i = 0; i < input.length; i++) {
            ret[i] = ret[i]/sum;
        }
        return ret;
    }

    // Feedfoward the input through layer, return output
    public Double[] feedfoward (Double[] input) {
        assert(input.length == numInput);
        Double[] summation = weighted_input(input);
        Double[] activations = activate(summation);
        return activations;
    }

    // Update weight using regularization parameter
    // the equation is w = (1 - eta*lambda/n) * w - weightsUpdate
    // regularization is eta*lambda/n
    public void updateWeights (Double[][] weightsUpdate, Double regularization) {
        assert(weights.length == weightsUpdate.length);
        for (int i = 0; i<momentum.length; i++) {
            assert(weights[i].length == weightsUpdate[i].length);
            for (int j = 0; j<momentum[i].length; j++) {
                momentum[i][j] = momentum_rate*momentum[i][j] - weightsUpdate[i][j] - regularization*weights[i][j];
                weights[i][j] += momentum[i][j];
            }
        }
    }

    // Update bias
    public void updateBias (Double[] biasUpdate) {
        assert(biasUpdate.length == bias.length);
        for (int i = 0; i<biasUpdate.length; i++) {
            bias[i] -= biasUpdate[i];
        }
    }

    // Calculate weight update matrix
    public Double[][] calculateWeightUpdate (Double[] delta_error, Double[] input) {
        assert(input.length == numInput);
        assert(delta_error.length == numOutput);
        Double[][] weightUpdate = new Double[delta_error.length][];
        for (int i = 0; i < delta_error.length; i++) {
            weightUpdate[i] = new Double[input.length];
            for (int j = 0; j < input.length; j++) {
                weightUpdate[i][j] = delta_error[i] * input[j];
            }
        }
        return weightUpdate;
    }

    // Calculate bias update matrix
    public Double[] calculateBiasUpdate (Double[] delta_error) {
        assert(delta_error.length == numOutput);
        return delta_error.clone();
    }

    // Calculate delta from previous layer weighted delta and this layer input
    public Double[] calculateDelta(Double[] weighted_delta, Double[] input) {
        assert(input.length == numInput);
        assert(weighted_delta.length == numOutput);
        throw new RuntimeException("Softmax Layer should used as output layer");
    }

    // Calculate error in output layer if it's last layer
    public Double[] calculateOutputError(Double[] input, Double[] target) {
        // Using Log Likelihood Function
        assert (input.length == numInput);
        assert (target.length == numOutput);
        Double[] output = feedfoward(input);
        Double[] delta = new Double[numOutput];
        for (int j = 0; j < output.length; j++) {
            delta[j] = (output[j] - target[j]);
        }
        return delta;
    }

    // Calculate the weighted delta (delta . weights) for next layer delta
    public Double[] calculateWeightedDelta(Double[] delta) {
        assert (delta.length == numOutput);
        Double[] weighted_delta = new Double[numInput];
        for (int j = 0; j < numInput; j++) {
            weighted_delta[j] = 0.0;
        }
        for (int i = 0; i < numOutput; i++) {
            Double[] weight = weights[i];
            for (int j = 0; j < numInput; j++) {
                weighted_delta[j] += weight[j] * delta[i];
            }
        }
        return weighted_delta;
    }

}
