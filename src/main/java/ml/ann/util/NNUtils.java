package ml.ann.util;

import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;

/**
 * A collection of utility methods for neural networks
 * Created by Alvin Natawiguna on 11/23/2015.
 */
public class NNUtils {

    public static double[] instancesToTargetVector (Instances instances, boolean signed) throws Exception {
        assert(instances != null);

        double[] ret = new double[instances.numInstances()];
        if (signed) {
            if (instances.classAttribute().isNumeric()) {
                throw new Exception("Numeric instances should be converted to binary first.");
            }

            int numValues = instances.classAttribute().numValues();
            if (numValues > 2) {
                throw new Exception("Signed instances can only have 2 class values.");
            }

            // convention: -1 for the first value, 1 for the second value
            for (int i = 0; i < instances.numInstances(); i++) {
                Instance instance = instances.instance(i);
                ret[i] = instance.classIndex() == 0 ? -1 : 1;
            }
        } else {
            for (int i = 0; i < instances.numInstances(); i++) {
                Instance instance = instances.instance(i);
                ret[i] = instance.classIndex();
            }
        }

        return ret;
    }

    public static double[] instancesToTargetVector (Instances instances) throws Exception {
        return instancesToTargetVector(instances, false);
    }

    /**
     * Produces a vector of targets from data instances
     *
     * @param instance the instances to be classified
     * @return vector of target instances
     */
    public static double[] instanceToTargetVector (Instance instance) {
        assert(instance != null);
        assert(!instance.classIsMissing());

        boolean isNominal = instance.classAttribute().isNominal();
        double[] ret;
        if (isNominal) {
            // this converts the nominal classes into a binary vector
            ret = new double[instance.classAttribute().numValues()];
            Arrays.fill(ret, 0.0);

            ret[(int)instance.classValue()] = 1.0;
        }
        else {
            ret = new double[1];
            ret[0] = instance.classValue();
        }

        return ret;
    }

    public static double[] instanceToInputVector (Instance instance) {
        double[] ret = new double[instance.numAttributes() - 1];

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
