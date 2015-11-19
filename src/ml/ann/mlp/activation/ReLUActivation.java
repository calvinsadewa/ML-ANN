package ml.ann.mlp.activation;

/**
 * Created by calvin-pc on 11/19/2015.
 */
public class ReLUActivation implements ActivationStrategy{
    @Override
    public double apply(Double x) {
        if (x > 0) return x;
        else return 0;
    }

    @Override
    public double derivation(Double x) {
        if (x > 0) return 1;
        else return 0;
    }
}
