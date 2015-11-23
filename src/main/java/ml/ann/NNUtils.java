package ml.ann;

import weka.core.Instance;

import java.util.Arrays;

/**
 * A collection of utility methods for neural networks
 * Created by Alvin Natawiguna on 11/23/2015.
 */
public class NNUtils {

    /**
     * Produces a vector of targets from data instances
     *
     * @param instance the instances to be classified
     * @return vector of target instances
     */
    public static double[] instanceToTargetVector (Instance instance) {
        assert(instance != null);

        boolean isNominal = instance.attribute(instance.classIndex()).isNominal();

        double[] ret;
        if (isNominal) {
            ret = new double[instance.classAttribute().numValues()];
            Arrays.fill(ret, 0.0);

            // TODO: what is this?
            ret[(int)instance.classValue()] = 1.0;
        }
        else {
            ret = new double[1];
            ret[0] = instance.classValue();
        }

        return ret;
    }

    public static double[] instanceToInputVector (Instance instance) {
        double[] ret;
        if (instance.classIsMissing()) {
            ret = new double[instance.numAttributes()];
        } else {
            ret = new double[instance.numAttributes() - 1];
        }

        int visitClass = 0;
        for (int i = 0; i < instance.numAttributes(); i++) {
            if (i == instance.classIndex()) {
                visitClass = 1;
            }
            else {
                ret[i - visitClass] = instance.value(i);
            }
        }
        return ret;
    }
}
