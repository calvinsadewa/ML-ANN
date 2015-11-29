package ml.ann.mlp.activation;

/**
 * Created by calvin-pc on 11/29/2015.
 */
public class LinearActivation implements ActivationStrategy{
    @Override
    public double apply(Double x) {
        return x;
    }

    @Override
    public double derivation(Double x) {
        return 1;
    }
}
