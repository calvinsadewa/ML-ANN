package ml.ann.mlp.weight;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Created by calvin-pc on 11/19/2015.
 */
public class RandomWeightAssignment implements WeightAssignmentStrategy {
    Random rnd = ThreadLocalRandom.current();
    @Override
    public double getWeight() {
        return rnd.nextGaussian();
    }
}
