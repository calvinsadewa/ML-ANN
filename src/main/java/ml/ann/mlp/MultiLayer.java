package ml.ann.mlp;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Created by calvin-pc on 11/18/2015.
 */
public class MultiLayer implements Serializable {
    public List<RegularizedMomentumLayer> layers = new ArrayList<>();
    public double learningRate = 0.1;
    public double regularizationRate = 0;
    public double momentumRate = 0;
    public int mini_batch_size = 1;
    public int epochs = 1000;
    public Double minDeltaMSE = 0.0;

    public Double[] feedfoward (Double[] input) {
        Double[] current_input = input.clone();
        for (RegularizedMomentumLayer l: layers) {
            current_input = l.feedfoward(current_input);
        }
        return current_input;
    }

    public void update_mini_batch(Double[][] input_data,
                                  Double[][] target_data,
                                  double learning_rate,
                                  double regularization_rate,
                                  int training_size) {

        // cumulative weight update by layer
        List<Double[][]> cumulativeWeightUpdateLayer = new ArrayList<>();
        List<Double[]> cumulativeBiasUpdateLayer = new ArrayList<>();

        //Calculate update by backprop
        for (RegularizedMomentumLayer l : layers) {
            Double[] bias = new Double[l.getNumOutput()];
            Double[][] weight_vec = new Double[l.getNumOutput()][];
            Arrays.fill(bias, 0.0);
            for (int j = 0; j < l.getNumOutput(); j++) {
                weight_vec[j] = new Double[l.getNumInput()];
                Arrays.fill(weight_vec[j], 0.0);
            }
            cumulativeWeightUpdateLayer.add(weight_vec);
            cumulativeBiasUpdateLayer.add(bias);
        }

        for (int i = 0; i < input_data.length; i++) {
            //Input by CustomActivationLayer
            List<Double[]> inputLayer = new ArrayList<>();

            Double[] target = target_data[i];
            Double[] current_input = input_data[i];
            for (RegularizedMomentumLayer l: layers) {
                inputLayer.add(current_input.clone());
                current_input = l.feedfoward(current_input);
            }

            // weight update by layer
            List<Double[][]> weightUpdateLayer = new ArrayList<>();
            List<Double[]> biasUpdateLayer = new ArrayList<>();

            // the delta dot weight matrix, used to propagate error to next layer
            Double[] weighted_delta = null;

            {
                RegularizedMomentumLayer current_layer = layers.get(layers.size() - 1);
                Double[] input = inputLayer.get(layers.size() - 1);
                Double[] delta = current_layer.calculateOutputError(input,target);
                weightUpdateLayer.add(current_layer.calculateWeightUpdate(delta,input));
                biasUpdateLayer.add(current_layer.calculateBiasUpdate(delta));
                weighted_delta = current_layer.calculateWeightedDelta(delta);
            }

            for (int j = layers.size() - 2; j >= 0; j--) {
                RegularizedMomentumLayer current_layer = layers.get(j);
                Double[] input = inputLayer.get(j);
                Double[] delta = current_layer.calculateDelta(weighted_delta,input);
                weightUpdateLayer.add(current_layer.calculateWeightUpdate(delta,input));
                biasUpdateLayer.add(current_layer.calculateBiasUpdate(delta));
                weighted_delta = current_layer.calculateWeightedDelta(delta);
            }

            Collections.reverse(weightUpdateLayer);
            Collections.reverse(biasUpdateLayer);

            //Add to cumulative update
            for (int j = 0; j < weightUpdateLayer.size(); j++) {
                Double [][] weightUpdate = weightUpdateLayer.get(j);
                Double [] biasUpdate = biasUpdateLayer.get(j);
                Double [][] cumWeightUpdate = cumulativeWeightUpdateLayer.get(j);
                Double [] cumBiasUpdate = cumulativeBiasUpdateLayer.get(j);
                for (int k = 0; k < weightUpdate.length; k++) {
                    cumBiasUpdate[k] += biasUpdate[k];
                    for (int l = 0; l < weightUpdate[k].length; l++) {
                        cumWeightUpdate[k][l] += weightUpdate[k][l];
                    }
                }
            }
        }

        // Update the network
        double regularization= learning_rate*regularization_rate/training_size;
        double ampilfy = learning_rate/input_data.length;
        for (int i = 0; i<layers.size(); i++) {
            Double[][] weightUpdate = cumulativeWeightUpdateLayer.get(i);
            Double[] biasUpdate = cumulativeBiasUpdateLayer.get(i);
            for (int j = 0; j< weightUpdate.length; j++) {
                biasUpdate[j] *= ampilfy;
                for (int k = 0; k< weightUpdate[j].length; k++) {
                    weightUpdate[j][k] *= ampilfy;
                }
            }

            layers.get(i).updateWeights(weightUpdate,regularization);
            layers.get(i).updateBias(biasUpdate);
        }
    }

    public void SGD (Double[][] input_data,
                     Double[][] target_data) {
        int current_epoch = 0;
        Double prevMSE = 0.0;
        Double currentDeltaMSE = Double.MAX_VALUE;

        for (RegularizedMomentumLayer l: layers) {
            l.setMomentum(momentumRate);
        }
        while (current_epoch < epochs && currentDeltaMSE > minDeltaMSE) {
            shuffleArray(input_data,target_data);
            for (int i = 0; i< input_data.length / mini_batch_size; i++) {
                int batch_size = 0;
                if (i+1 == input_data.length / mini_batch_size) batch_size = input_data.length;
                else batch_size = i* mini_batch_size + mini_batch_size;
                Double[][] input_training = Arrays.copyOfRange(input_data, i * mini_batch_size, batch_size);
                Double[][] target_training = Arrays.copyOfRange(target_data,i*mini_batch_size, batch_size);
                update_mini_batch(input_training,target_training,learningRate,regularizationRate,input_data.length);
            }

            current_epoch ++;
            Double currentMSE = calculateMSE(input_data,target_data);
            currentDeltaMSE = Math.abs(currentMSE - prevMSE);
            prevMSE = currentMSE;
            System.out.println("MSE = " + prevMSE);
        }
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

    // Implementing Fisher-Yates shuffle
    static<T> void shuffleArray(T[] ar, T[] ar2)
    {
        // If running on Java 6 or older, use `new Random()` on RHS here
        Random rnd = ThreadLocalRandom.current();
        for (int i = ar.length - 1; i > 0; i--)
        {
            int index = rnd.nextInt(i + 1);
            // Simple swap
            T a = ar[index];
            ar[index] = ar[i];
            ar[i] = a;
            a = ar2[index];
            ar2[index] = ar2[i];
            ar2[i] = a;
        }
    }

    public RegularizedMomentumLayer getLastLayer() {
        return layers.get(layers.size() - 1);
    }

}
