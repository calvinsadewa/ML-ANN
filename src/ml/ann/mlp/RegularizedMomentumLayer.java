package ml.ann.mlp;

/**
 * Created by calvin-pc on 11/19/2015.
 */
public interface RegularizedMomentumLayer {

    // Get number of input
    public int getNumInput();

    // Get number of output
    public int getNumOutput();

    public void setMomentum(Double momentum);
    public double getMomentum();

    // return output of this layer
    public Double[] feedfoward (Double[] input);

    // Update this weight layer
    public void updateWeights (Double[][] weightsUpdate, Double regularization);

    // Update this bias layer
    public void updateBias (Double[] biasUpdate);

    // Calculate weight update
    public Double[][] calculateWeightUpdate (Double[] delta_error, Double[] input);

    // Calculate bias update
    public Double[] calculateBiasUpdate (Double[] delta_error);

    // Calculate delta by using previous layer weighted delta
    public Double[] calculateDelta(Double[] weighted_delta, Double[] input);

    // Calculate delta of ouput layer
    public Double[] calculateOutputError(Double[] input, Double[] target);

    // Calculate weighted delta for next backpropagation
    public Double[] calculateWeightedDelta(Double[] delta);
}
