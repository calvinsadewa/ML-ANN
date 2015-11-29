package ml.ann.mlp;

import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.supervised.attribute.NominalToBinary;

import java.io.Serializable;

/**
 * Created by calvin-pc on 11/29/2015.
 */
public class NominalSLPClassifier implements Classifier,Serializable {

    FilteredClassifier m_root = null;

    public NominalSLPClassifier(SLPClassifier slc) {
        this.m_root = new FilteredClassifier();
        m_root.setFilter(new NominalToBinary());
        m_root.setClassifier(slc);
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        m_root.buildClassifier(data);
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception{
        return m_root.classifyInstance(instance);
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        return m_root.distributionForInstance(instance);
    }

    @Override
    public Capabilities getCapabilities() {
        return null;
    }
}
