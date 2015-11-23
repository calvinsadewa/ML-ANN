package ml.ann.single_node;

import ml.ann.weight.GivenWeightAssignment;
import ml.ann.weight.RandomWeightAssignment;
import ml.ann.weight.WeightAssignmentStrategy;
import ml.ann.NNUtils;

import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by Alvin Natawiguna on 11/23/2015.
 */
public class DeltaRuleIncremental extends SingleNodeClassifier {


    @Override
    protected void classify(double[][] inputVectors, double[][] targetVectors) {

    }
}
