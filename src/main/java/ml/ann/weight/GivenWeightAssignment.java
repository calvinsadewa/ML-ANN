package ml.ann.weight;

/**
 * Created by calvin-pc on 11/19/2015.
 */
public class GivenWeightAssignment implements WeightAssignmentStrategy{
    Double given;
    public GivenWeightAssignment(Double n) {
        given = n;
    }

    @Override
    public double getWeight() {
        return given;
    }
}
