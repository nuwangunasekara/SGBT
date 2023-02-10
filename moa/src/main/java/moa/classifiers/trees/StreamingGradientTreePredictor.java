package moa.classifiers.trees;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.stream.IntStream;

import com.github.javacliparser.*;
import com.henrygouk.sgt.*;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.Instance;

import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.*;
import moa.classifiers.trees.sgt.*;
import moa.core.Measurement;

public class StreamingGradientTreePredictor extends BoostingCommittee implements Serializable, MultiClassClassifier, Regressor, SemiSupervisedLearner {

    private static final long serialVersionUID = 1L;

    protected MultiOutputLearner mTrees;

    protected AttributeDiscretizer mDiscretizer;

    protected int mInstances;

    protected Objective mObjective;

    public FloatOption delta = new FloatOption("delta", 'D',
            "The confidence level used when performing the hypothesis tests.", 1E-7, 0.0, 1.0);

    public FloatOption lambda = new FloatOption("lambda", 'L',
            "Regularisation parameter that can be used to influence the magnitude of updates.", 0.1, 0.0, Double.POSITIVE_INFINITY);

    public FloatOption gamma = new FloatOption("gamma", 'Y',
            "The loss incurred from adding a new node to the tree.", 1.0, 0.0, Double.POSITIVE_INFINITY);

    public IntOption gracePeriod = new IntOption("gracePeriod", 'G',
            "The number of instances to observe between searches for new splits.", 200, 0, Integer.MAX_VALUE);

    public IntOption warmStart = new IntOption("warmStart", 'W',
            "The number of instances used to estimate bin boundaries for numeric values.", 1000, 0, Integer.MAX_VALUE);

    public IntOption bins = new IntOption("bins", 'B',
            "The number of bins to be used for discretizing numeric attributes.", 64, 0, Integer.MAX_VALUE);

    public FloatOption semiSupervisedOption = new FloatOption("enableSemiSupervised", 'U',
            "Enables learning from unlabelled instances", 0.0, 0.0, 1.0);

    @Override
    public String getPurposeString() {
        return "Trains a single Streaming Gradient Tree for regression, or a committe for classification.";
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    @Override
    public void resetLearningImpl() {
        System.out.println("Resetting SGT");
        mTrees = null;
        mDiscretizer = new AttributeDiscretizer(bins.getValue());
        mInstances = 0;
    }

    @Override
    public long measureByteSize() {
        return 0;
    }

    protected MultiOutputLearner createTrees(FeatureInfo[] featureInfo, StreamingGradientTreeOptions options, int numOutputs) {
        return new StreamingGradientTreeCommittee(featureInfo, options, numOutputs);
    }
    public void trainOnInstanceImpl(Instance[] instances, int multipleIterationByHessianCeiling, GradHess[] gradHess){
            String currentMethod = new Exception().getStackTrace()[0].getMethodName();
        throw new UnsupportedOperationException(this.getClass().getName() + " " +currentMethod);
    }

    public void trainOnInstanceImpl(Instance inst, GradHess[] gradHess, double[] raw /* only for semiSupervisedOption */) {
        mInstances++;

        if(mInstances <= warmStart.getValue()) {
            mDiscretizer.observe(inst);
            return;
        }

        Attribute target = inst.classAttribute();

        if(mTrees == null) {
            FeatureInfo[] featureInfo = mDiscretizer.getFeatureInfo();
            StreamingGradientTreeOptions options = new StreamingGradientTreeOptions();
            options.delta = delta.getValue();
            options.gracePeriod = gracePeriod.getValue();
            options.lambda = lambda.getValue();
            options.gamma = gamma.getValue();

            if(target.isNominal()) {
                mTrees = createTrees(featureInfo, options, target.numValues() - 1);
                committeeSize = target.numValues() - 1;
//                mObjective = new SoftmaxCrossEntropy();
            }
            else {
                mTrees = createTrees(featureInfo, options, 1);
                committeeSize = 1;
//                mObjective = new SquaredError();
            }
        }

        int[] features = mDiscretizer.getFeatures(inst);
        double[] groundTruth;
//        double[] raw = mTrees.predict(features);

        if(target.isNominal()) {
            if(!inst.classIsMissing()) {
                groundTruth = new double[target.numValues()];
                groundTruth[(int)inst.classValue()] = 1.0;
            }
            else if(semiSupervisedOption.getValue() > 0.0) {
                //This is equivalent to entropy minimisation when mObjective is the SoftmaxCrossEntropy objective
                groundTruth = mObjective.transfer(raw);

                for(int j = 0; j < groundTruth.length; j++) {
                    groundTruth[j] *= semiSupervisedOption.getValue();
                }
            }
            else {
                return;
            }
        }
        else {
            groundTruth = new double[] {inst.classValue()};
        }

//        GradHess[] gradHess = mObjective.computeDerivatives(groundTruth, raw);
        Double[] weights = IntStream.range(0, gradHess.length)
//                .parallel()
                .mapToObj(i -> inst.weight()).toArray(Double[]::new);
        mTrees.update(features, gradHess, weights);
    }

//    @Override
//    public HashMap getCommitteeInfo() {
//        HashMap<Integer, ArrayList> committeeInformation=new HashMap<Integer,ArrayList>();//Creating HashMap
//        ArrayList<String> treeInformation = new ArrayList<>();
//        int n = mTrees != null ? mTrees.getNumNodes(): 0;
//        String s =".";
//        String sizeString = new String(new char[n]).replace("\0", s);
////                treeInformation.put(" ", sizeString);
//        treeInformation.add(sizeString);
//        committeeInformation.put(0,treeInformation);
//        return committeeInformation;
//    }

    @Override
    public ArrayList<ArrayList<HashMap<String,String>>> getCommitteeInfo() {
        if (mTrees == null){
            return null;
        }
        ArrayList<ArrayList<HashMap<String,String>>> committeeInformation=new ArrayList<ArrayList<HashMap<String,String>>>();//Creating HashMap
        for(int i = 0; i < ((StreamingGradientTreeCommittee) mTrees).mTrees.length; i++) {
            ArrayList<HashMap<String,String>> treeInformation1 = new ArrayList<HashMap<String,String>>();
            HashMap<String,String> hashMap=new HashMap<String,String>();
            int n = 0;
            String treeType = "U:";
            if (((StreamingGradientTreeCommittee) mTrees).mTrees[i] instanceof StreamingGradientTree){
                n = ((StreamingGradientTreeCommittee) mTrees).mTrees[i] != null ?
                        ((StreamingGradientTreeCommittee) mTrees).mTrees[i].getNumNodes(): 0;
                treeType = "S:";
            }
//            String s =".";
            String sizeString = treeType;
//            sizeString += new String(new char[n]).replace("\0", s); // put n s into sizeString
//            treeInformation.add(sizeString);
            hashMap.put("type", treeType);
            hashMap.put("numNodes", "" + n);
//            committeeInformation.add(treeInformation);
            committeeInformation.add(treeInformation1);
        }

        return committeeInformation;
    }


    public void trainOnInstanceImpl(Instance inst) {
        mInstances++;

        if(mInstances <= warmStart.getValue()) {
            mDiscretizer.observe(inst);
            return;
        }

        Attribute target = inst.classAttribute();

        if(mTrees == null) {
            FeatureInfo[] featureInfo = mDiscretizer.getFeatureInfo();
            StreamingGradientTreeOptions options = new StreamingGradientTreeOptions();
            options.delta = delta.getValue();
            options.gracePeriod = gracePeriod.getValue();
            options.lambda = lambda.getValue();
            options.gamma = gamma.getValue();

            if(target.isNominal()) {
                mTrees = createTrees(featureInfo, options, target.numValues() - 1);
                committeeSize = target.numValues() - 1;
                mObjective = new SoftmaxCrossEntropy();
            }
            else {
                mTrees = createTrees(featureInfo, options, 1);
                committeeSize = 1;
                mObjective = new SquaredError();
            }
        }

        int[] features = mDiscretizer.getFeatures(inst);
        double[] groundTruth;
        double[] raw = mTrees.predict(features);

        if(target.isNominal()) {
            if(!inst.classIsMissing()) {
                groundTruth = new double[target.numValues()];
                groundTruth[(int)inst.classValue()] = 1.0;
            }
            else if(semiSupervisedOption.getValue() > 0.0) {
                //This is equivalent to entropy minimisation when mObjective is the SoftmaxCrossEntropy objective
                groundTruth = mObjective.transfer(raw);

                for(int j = 0; j < groundTruth.length; j++) {
                    groundTruth[j] *= semiSupervisedOption.getValue();
                }
            }
            else {
                return;
            }
        }
        else {
            groundTruth = new double[] {inst.classValue()};
        }

        GradHess[] gradHess = mObjective.computeDerivatives(groundTruth, raw, true, false);
        Double[] weights = IntStream.range(0, gradHess.length)
//                .parallel()
                .mapToObj(i -> inst.weight()).toArray(Double[]::new);
        mTrees.update(features, gradHess, weights);
    }



    public double[] getScoresForInstance(Instance inst) {
        if(mTrees == null) {
            return getScoresWhenNullTree(committeeSize);
        }

        int[] features = mDiscretizer.getFeatures(inst);
        return mTrees.predict(features);
    }

    public double[] getVotesForInstance(Instance inst) {
        if(mTrees == null) {
            return getScoresWhenNullTree(inst.classAttribute().isNominal() ?committeeSize + 1 : 1);
        }
        double[] raw = getScoresForInstance(inst);
//        return mObjective.transfer(raw);
        return inst.classAttribute().isNominal() ?  mObjective.transfer(raw) : raw;
    }

    @Override
    public void getModelDescription(StringBuilder in, int indent) {
        //
    }

    @Override
    public Measurement[] getModelMeasurementsImpl() {
        if(true){
            return null;
        }

        double nodes = 0.0;
        double splits = 0.0;
        double updates = 0.0;
        double maxDepth = 0.0;

        if(mTrees != null) {
            nodes = mTrees.getNumNodes();
            splits = mTrees.getNumSplits();
            updates = mTrees.getNumNodeUpdates();
            maxDepth = mTrees.getMaxDepth();
        }

        return new Measurement[] {
                new Measurement("nodes", nodes),
                new Measurement("splits", splits),
                new Measurement("node updates", updates),
                new Measurement("max depth", maxDepth)
        };
    }

    @Override
    public ImmutableCapabilities defineImmutableCapabilities() {
        if (this.getClass() == StreamingGradientTreePredictor.class)
            return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
        else
            return new ImmutableCapabilities(Capability.VIEW_STANDARD);
    }
}
