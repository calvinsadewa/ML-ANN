package ml.ann.mlp.weight;

import java.io.Serializable;

/**
 * Created by calvin-pc on 11/19/2015.
 */
public interface WeightAssignmentStrategy extends Serializable {
    public double getWeight();
}
