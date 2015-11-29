import org.junit.Test;
import static org.junit.Assert.*;

import ml.ann.single_node.SingleNodeNN;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import java.io.InputStream;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Created by Alvin Natawiguna on 11/29/2015.
 */
public class SingleNodeNNTest {


    @Test
    public void testPerceptronNominalValue() {
        InputStream is = this.getClass().getResourceAsStream("weather.nominal.arff");
        assert (is != null);

        DataSource data = null;
        Instances testData = null;
        try {
            data = new DataSource(is);
            testData = data.getDataSet();
            testData.setClassIndex(testData.numAttributes() - 1);

            Classifier classifier = new SingleNodeNN();
            Evaluation evaluation = new Evaluation(testData);

            evaluation.crossValidateModel(classifier, testData, 10, ThreadLocalRandom.current());
            System.out.println(evaluation.toSummaryString("Perceptron with nominal value result:", false));
        } catch (Exception e) {
            e.printStackTrace();
            assertTrue("Classifier failed!", false);
        }
        assertNotNull(data);
        assertNotNull(testData);
    }

    @Test
    public void testPerceptronNumericValue() {

    }

    @Test
    public void testPerceptronNominalClass() {

    }
}
