package ml.ann.mlp.activation;

import java.io.Serializable;

/**
 * Created by calvin-pc on 11/17/2015.
 */
// Interface for activation function like sigmoid or tanh
public interface ActivationStrategy extends Serializable {
    //Apply x to activation function
    double apply(Double x);
    //Apply x to derivation of activation function
    double derivation(Double x);
}
