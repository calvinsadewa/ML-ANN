package ml.ann.mlp.activation;

/**
 * Created by calvin-pc on 11/17/2015.
 */
public class SigmoidActivation implements ActivationStrategy {
    @Override
    public double apply(Double x) {
        return 1.0/(1.0+Math.exp(-x));
    }

    @Override
    public double derivation(Double x) {
        Double sig = this.apply(x);
        return sig * (1-sig);
    }
}
