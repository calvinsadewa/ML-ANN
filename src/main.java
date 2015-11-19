/**
 * Created by calvin-pc on 9/25/2015.
 */

import ml.ann.mlp.CrossEntrophySigmoidLayer;
import ml.ann.mlp.MLPClassifier;
import ml.ann.mlp.MultiLayer;
import ml.ann.mlp.NominalMLPClassifier;
import ml.ann.mlp.weight.RandomWeightAssignment;
import ml.ann.mlp.weight.WeightAssignmentStrategy;
import weka.classifiers.trees.j48.*;
public class main {
    public static void main(String[] args) throws Exception{
        float cf = 0.5f;
        WekaAccessor wa = new WekaAccessor();
        wa.loadData("D:\\Program Files\\Weka-3-7\\data\\iris.arff");
        wa.supervisedResample();
        MLPClassifier mlc = new MLPClassifier();
        MultiLayer ml = new MultiLayer();
        ml.mini_batch_size = 20;
        ml.momentumRate = 0.2;
        ml.regularizationRate = 0.4;
        WeightAssignmentStrategy ws = new RandomWeightAssignment();
        ml.layers.add(new CrossEntrophySigmoidLayer(10,10,ws));
        mlc.ml = ml;
        wa.classifier = new NominalMLPClassifier(mlc);
        wa.classifier.buildClassifier(wa.data);
        double[] hasil = wa.test(wa.data);
        System.out.println(wa.evaluation.toSummaryString());
    }
}