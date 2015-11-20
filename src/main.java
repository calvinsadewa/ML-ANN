/**
 * Created by calvin-pc on 9/25/2015.
 */

import ml.ann.mlp.*;
import ml.ann.mlp.activation.ReLUActivation;
import ml.ann.mlp.activation.SigmoidActivation;
import ml.ann.mlp.weight.RandomWeightAssignment;
import ml.ann.mlp.weight.WeightAssignmentStrategy;
import weka.classifiers.trees.j48.*;
public class main {
    public static void main(String[] args) throws Exception{
        float cf = 0.5f;
        WekaAccessor wa = new WekaAccessor();
        wa.loadData("D:\\Program Files\\Weka-3-7\\data\\iris.2D.arff");
        wa.supervisedResample();
        MLPClassifier mlc = new MLPClassifier();
        MultiLayer ml = new MultiLayer();
        ml.mini_batch_size = 1;
        ml.momentumRate = 0.8;
        ml.regularizationRate = 0.1;
        ml.epochs = 1000;
        ml.learningRate = 0.01;
        ml.minDeltaMSE = 0.0;
        WeightAssignmentStrategy ws = new RandomWeightAssignment();
        // tambah hidden layer
        ml.layers.add(new CustomActivationQuadraticCostLayer(20,12,new SigmoidActivation(),ws));
        mlc.baseMl = ml;
        mlc.use_SigmoidOutput = true;
        mlc.use_CrossEnthropyCost = false;
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