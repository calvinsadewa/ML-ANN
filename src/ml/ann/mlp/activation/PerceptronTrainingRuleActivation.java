package ml.ann.mlp.activation;

/**
 * Created by calvin-pc on 11/29/2015.
 */
// Perceptron training rule activation with output 1 or 0
public class PerceptronTrainingRuleActivation implements ActivationStrategy {
    @Override
    public double apply(Double x) {
        if (x > 0) return 1;
        else return 0;
    }

    @Override
    public double derivation(Double x) {
        return 1;
    }
}
