package ml.ann.mlp;

import ml.ann.mlp.activation.ActivationStrategy;
import ml.ann.mlp.util.DeepCopy;
import ml.ann.mlp.weight.WeightAssignmentStrategy;
import weka.classifiers.AbstractClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.io.Serializable;
import java.util.Arrays;

/**
 * Created by calvin-pc on 11/29/2015.
 */
public class SLPClassifier extends AbstractClassifier implements Serializable {
    public ActivationStrategy activation = null;
    private CustomActivationQuadraticCostLayer layer;
    public WeightAssignmentStrategy ws = null;
    public double learningRate = 0.1;
    public int epochs = 1000;
    public Double minDeltaMSE = 0.0;
    public boolean batch = false;
    public Double minMSE = 0.0;
    private boolean m_Nominal = false;

    public Double[] feedfoward (Double[] input) {
        Double[] current_input = input.clone();
        return layer.feedfoward(current_input);
    }

    public Double calculateMSE(Double[][] input_data,
                               Double[][] target_data) {
        Double sum = 0.0;
        for (int i =0; i< input_data.length; i++) {
            Double cost = 0.0;
            Double[] input = input_data[i];
            Double[] target = target_data[i];
            Double[] output = this.feedfoward(input);

            for (int j = 0; j<target.length; j++) {
                cost = cost + Math.pow(target[j]-output[j],2);
            }
            cost = cost / (2 * Math.max(output.length,1));
            sum = cost + sum;
        }
        return sum;
    }

    private void update_mini_batch (Double[][] input_data,
                                    Double[][] target_data) {
        Double[] bias_update = new Double[layer.getNumOutput()];
        Double[][] weight_update = new Double[layer.getNumOutput()][];
        Arrays.fill(bias_update, 0.0);
        for (int j = 0; j<layer.getNumOutput(); j++) {
            weight_update[j] = new Double[layer.getNumInput()];
            Arrays.fill(weight_update[j],0.0);
        }

        for (int i = 0; i<input_data.length; i++) {
            Double[] input = input_data[i];
            Double[] output = input_data[i];
            Double[] error = layer.calculateOutputError(input,output);
            Double[][] weight_upd = layer.calculateWeightUpdate(error,input);
            Double[] bias_upd = layer.calculateBiasUpdate(error);

            for (int k = 0; k < bias_update.length; k++) {
                bias_update[k] += bias_upd[k] * learningRate;
            }
            for (int k = 0; k < weight_update.length; k++) {
                for (int j = 0; j < weight_update[k].length; j++) {
                    weight_update[k][j] += weight_upd[k][j] * learningRate;
                }
            }
        }

        layer.updateWeights(weight_update, 0.0);
        layer.updateBias(bias_update);
    }

    public void train (Double[][] input_data,
                       Double[][] target_data) {
        int current_epoch = 0;
        Double prevMSE = 0.0;
        Double currentDeltaMSE = Double.MAX_VALUE;
        Double currentMSE = Double.MAX_VALUE;

        while (current_epoch < epochs && currentDeltaMSE > minDeltaMSE && currentMSE > minMSE) {
            if (batch) {
                update_mini_batch(input_data,target_data);
            }
            else {
                for (int j = 0; j < input_data.length; j++) {
                    Double[][] input_train = new Double[1][];
                    Double[][] output_train = new Double[1][];
                    input_train[0] = input_data[j];
                    output_train[0] = target_data[j];
                    update_mini_batch(input_train, output_train);
                }
            }
            current_epoch ++;
            currentMSE = calculateMSE(input_data,target_data);
            currentDeltaMSE = Math.abs(currentMSE - prevMSE);
            prevMSE = currentMSE;
            System.out.println("MSE = " + prevMSE);
        }
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        Instances filtered = new Instances(data);
        filtered.deleteWithMissingClass();
        m_Nominal = data.attribute(data.classIndex()).isNominal();

        Double[][] input_data = new Double[filtered.size()][];
        Double[][] output_data = new Double[filtered.size()][];
        for (int i = 0; i < filtered.size(); i++) {
            input_data[i] = instanceToInputVector(filtered.get(i));
            output_data[i] = instanceToTargetVector(filtered.get(i));
        }
        layer = new CustomActivationQuadraticCostLayer(input_data[0].length,output_data[0].length,activation,ws);
        layer.momentum_rate = 0;

        train(input_data, output_data);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        Double[] input = instanceToInputVector(instance);
        Double[] output = layer.feedfoward(input);
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
        Double[] output = layer.feedfoward(input);
        double[] ret = new double[output.length];
        Double sum = 0.0;
        Double min = Double.MAX_VALUE;
        for (int i = 0; i< output.length; i++) {
            ret[i] = output[i].doubleValue();
            if (min > ret[i]) min = ret[i];
        }

        if (min < 0) {
            for (int i = 0; i < output.length; i++) {
                ret[i] -= min;
            }
        }
        for (int i = 0; i< output.length; i++) {
            sum += ret[i];
        }

        if (sum == 0) sum = 1.0;
        for (int i = 0; i< output.length; i++) {
            ret[i] /= sum;
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
            Arrays.fill(ret, 0.0);
            ret[(int)i.classValue()] = 1.0;
            return ret;
        }
        else {
            Double[] ret = new Double[1];
            ret[0] = i.classValue();
            return ret;
        }
    }
}
