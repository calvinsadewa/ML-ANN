/**
 * Created by calvin-pc on 9/25/2015.
 */

import ml.ann.mlp.*;
import ml.ann.mlp.activation.SigmoidActivation;
import ml.ann.weight.RandomWeightAssignment;
import ml.ann.weight.WeightAssignmentStrategy;

import java.io.InputStream;

public class Main {
    public static void main(String[] args) throws Exception{
        float confidenceRate = 0.5f;
        WekaAccessor wa = new WekaAccessor();

        // load sample ARFF from weka
        InputStream is = Main.class.getResourceAsStream("iris.2D.arff");

        wa.loadData(is);
        wa.supervisedResample();

        // initialize the class
        MLPClassifier mlc = new MLPClassifier();
        MultiLayer ml = new MultiLayer();

        // setup parameters and resampling size
        ml.mini_batch_size = 1;
        ml.momentumRate = 0.8;
        ml.regularizationRate = 0.1;
        ml.epochs = 1000;
        ml.learningRate = 0.01;
        ml.minDeltaMSE = 0.0;
        WeightAssignmentStrategy ws = new RandomWeightAssignment();

        // tambah hidden layer
        ml.layers.add(new CustomActivationQuadraticCostLayer(20, 12, new SigmoidActivation(), ws));
        mlc.baseMl = ml;
        mlc.use_SigmoidOutput = true;
        mlc.use_CrossEntropyCost = false;
        wa.classifier = new NominalMLPClassifier(mlc);

        // input layer dan output layer di generate otomatis
        wa.classifier.buildClassifier(wa.data);
        double[] hasil = wa.test(wa.data);
        String s = wa.evaluation.toSummaryString();
        wa.crossValidation();

        System.out.println(s);
        System.out.println(wa.evaluation.toSummaryString());
    }
}