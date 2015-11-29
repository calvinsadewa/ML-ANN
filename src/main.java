/**
 * Created by calvin-pc on 9/25/2015.
 */

import ml.ann.mlp.*;
import ml.ann.mlp.activation.LinearActivation;
import ml.ann.mlp.activation.PerceptronTrainingRuleActivation;
import ml.ann.mlp.activation.ReLUActivation;
import ml.ann.mlp.activation.SigmoidActivation;
import ml.ann.mlp.weight.RandomWeightAssignment;
import ml.ann.mlp.weight.WeightAssignmentStrategy;
import weka.classifiers.Classifier;
import weka.classifiers.trees.j48.*;
public class main {
    public static void main(String[] args) throws Exception{
        float cf = 0.5f;
        WekaAccessor wa = new WekaAccessor();
        wa.loadData("D:\\Program Files\\Weka-3-7\\data\\iris.arff");
        wa.supervisedResample();

        wa.classifier = getMLP();
        // input layer dan output layer di generate otomatis
        wa.classifier.buildClassifier(wa.data);
        wa.crossValidation();
        String MLP = wa.evaluation.toSummaryString();


        wa.classifier = getPTR();
        // input layer dan output layer di generate otomatis
        wa.classifier.buildClassifier(wa.data);
        wa.crossValidation();
        String PLTR = wa.evaluation.toSummaryString();

        wa.classifier = getDeltaBatch();
        // input layer dan output layer di generate otomatis
        wa.classifier.buildClassifier(wa.data);
        wa.crossValidation();
        String DeltaB = wa.evaluation.toSummaryString();

        wa.classifier = getDeltaIncr();
        // input layer dan output layer di generate otomatis
        wa.classifier.buildClassifier(wa.data);
        wa.crossValidation();
        String DeltaI = wa.evaluation.toSummaryString();

        System.out.println(MLP);
        System.out.println(DeltaB);
        System.out.println(DeltaI);
        System.out.println(PLTR);
    }

    public static Classifier getMLP() {
        MLPClassifier mlc = new MLPClassifier();
        MultiLayer ml = new MultiLayer();
        ml.mini_batch_size = 1;
        ml.momentumRate = 0.1;
        ml.regularizationRate = 0.0;
        ml.epochs = 1000;
        ml.learningRate = 0.01;
        ml.minDeltaMSE = 0.0;
        WeightAssignmentStrategy ws = new RandomWeightAssignment();
        // tambah hidden layer
        ml.layers.add(new CustomActivationQuadraticCostLayer(20,12,new SigmoidActivation(),ws));
        mlc.baseMl = ml;
        mlc.use_SigmoidOutput = false;
        mlc.use_CrossEnthropyCost = false;
        return new NominalMLPClassifier(mlc);
    }

    public static Classifier getPTR() {
        SLPClassifier mlc = new SLPClassifier();
        mlc.batch = false;
        mlc.activation = new PerceptronTrainingRuleActivation();
        mlc.ws = new RandomWeightAssignment();
        mlc.epochs = 1000;
        mlc.learningRate = 0.01;
        mlc.minDeltaMSE = 0.0;
        // tambah hidden layer
        return new NominalSLPClassifier(mlc);
    }

    public static Classifier getDeltaIncr() {
        SLPClassifier mlc = new SLPClassifier();
        mlc.batch = false;
        mlc.activation = new LinearActivation();
        mlc.ws = new RandomWeightAssignment();
        mlc.epochs = 1000;
        mlc.learningRate = 0.01;
        mlc.minDeltaMSE = 0.0;
        // tambah hidden layer
        return new NominalSLPClassifier(mlc);
    }

    public static Classifier getDeltaBatch() {
        SLPClassifier mlc = new SLPClassifier();
        mlc.batch = false;
        mlc.activation = new LinearActivation();
        mlc.ws = new RandomWeightAssignment();
        mlc.epochs = 1000;
        mlc.learningRate = 0.01;
        mlc.minDeltaMSE = 0.0;
        // tambah hidden layer
        return new NominalSLPClassifier(mlc);
    }
}