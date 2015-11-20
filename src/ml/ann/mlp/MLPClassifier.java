package ml.ann.mlp;

import com.sun.org.apache.xpath.internal.operations.Mult;
import ml.ann.mlp.activation.ReLUActivation;
import ml.ann.mlp.activation.SigmoidActivation;
import ml.ann.mlp.util.DeepCopy;
import ml.ann.mlp.weight.RandomWeightAssignment;
import ml.ann.mlp.weight.WeightAssignmentStrategy;
import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Created by calvin-pc on 11/19/2015.
 * Class for MLP classifier, where all input is numeric and the class can be nominal or numeric
 */
public class MLPClassifier extends AbstractClassifier implements Serializable{
    public MultiLayer ml = new MultiLayer();
    public MultiLayer baseMl = new MultiLayer();
    public WeightAssignmentStrategy ws = new RandomWeightAssignment();
    public boolean m_Nominal = true;
    public boolean use_SigmoidOutput = false;
    public boolean use_CrossEnthropyCost = false;

    @Override
    public void buildClassifier(Instances data) throws Exception {
        ml = (MultiLayer)DeepCopy.copy(baseMl);
        Instances filtered = new Instances(data);
        filtered.deleteWithMissingClass();
        m_Nominal = data.attribute(data.classIndex()).isNominal();

        Double[][] input_data = new Double[filtered.size()][];
        Double[][] output_data = new Double[filtered.size()][];
        for (int i = 0; i < filtered.size(); i++) {
            input_data[i] = instanceToInputVector(filtered.get(i));
            output_data[i] = instanceToTargetVector(filtered.get(i));
        }
        addInputLayer(input_data[0].length);
        addOutputLayer(output_data[0].length);

        ml.SGD(input_data,output_data);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        Double[] input = instanceToInputVector(instance);
        Double[] output = ml.feedfoward(input);
        if (m_Nominal) {
            int max = 0;
            for (int i = 0; i<output.length; i++) {
                if (output[i] > output[max]) max = i;
            }
            return new Double(max);
        }
        else {
            return output[0];
        }
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        Double[] input = instanceToInputVector(instance);
        Double[] output = ml.feedfoward(input);
        double[] ret = new double[output.length];
        for (int i = 0; i< output.length; i++) {
            ret[i] = output[i].doubleValue();
        }
        return ret;
    }

    @Override
    public Capabilities getCapabilities() {
        Capabilities result = super.getCapabilities();
        result.disableAll();

        // attributes
        //result.enable(Capabilities.Capability.NOMINAL_ATTRIBUTES);
        result.enable(Capabilities.Capability.NUMERIC_ATTRIBUTES);
        //result.enable(Capabilities.Capability.DATE_ATTRIBUTES);
        //result.enable(Capabilities.Capability.MISSING_VALUES);

        // class
        result.enable(Capabilities.Capability.NOMINAL_CLASS);
        result.enable(Capabilities.Capability.NUMERIC_CLASS);
        //result.enable(Capabilities.Capability.DATE_CLASS);
        result.enable(Capabilities.Capability.MISSING_CLASS_VALUES);

        return result;
    }

    private Double[] instanceToInputVector (Instance i) {
        Double[] ret = new Double[i.numAttributes() - 1];
        if (i.classIsMissing()) ret = new Double[i.numAttributes()];
        int visitClass = 0;
        for (int j = 0; j< i.numAttributes(); j++) {
            if (j == i.classIndex()) {visitClass = 1;}
            else {
                ret[j - visitClass] = i.value(j);
            }
        }
        return ret;
    }

    private Double[] instanceToTargetVector (Instance i) {
        if (m_Nominal) {
            Double[] ret = new Double[i.classAttribute().numValues()];
            Arrays.fill(ret,0.0);
            ret[(int)i.classValue()] = 1.0;
            return ret;
        }
        else {
            Double[] ret = new Double[1];
            ret[0] = i.classValue();
            return ret;
        }
    }

    // Call this last
    private void addOutputLayer(int numOuput) {
        if (use_SigmoidOutput) {
            if (use_CrossEnthropyCost)
                ml.layers.add(new CrossEntrophySigmoidLayer(ml.getLastLayer().getNumOutput(),numOuput,ws));
            else
                ml.layers.add(new CustomActivationQuadraticCostLayer(ml.getLastLayer().getNumOutput(),numOuput, new SigmoidActivation(),ws));
        }
        else if (m_Nominal) {
            ml.layers.add(new SoftmaxLayer(ml.getLastLayer().getNumOutput(),numOuput,ws));
        }
        else {
            ml.layers.add(new CustomActivationQuadraticCostLayer(ml.getLastLayer().getNumOutput(),numOuput, new ReLUActivation(),ws));
        }
    }

    // Call this first
    private void addInputLayer(int numInput) {
        if (m_Nominal) {
            if (ml.layers.size() == 0) ml.layers.add(new CrossEntrophySigmoidLayer(numInput,numInput,ws));
            else ml.layers.add(0,new CrossEntrophySigmoidLayer(numInput,ml.layers.get(0).getNumInput(),ws));
        }
        else {
            if (ml.layers.size() == 0) ml.layers.add(new CustomActivationQuadraticCostLayer(numInput,numInput,new ReLUActivation(),ws));
            else ml.layers.add(0,new CustomActivationQuadraticCostLayer(numInput,ml.layers.get(0).getNumInput(),new ReLUActivation(),ws));
        }
    }
}
