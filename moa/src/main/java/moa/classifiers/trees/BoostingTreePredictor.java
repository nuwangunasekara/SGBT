package moa.classifiers.trees;

import com.henrygouk.sgt.*;
import com.yahoo.labs.samoa.instances.Instance;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.*;
import moa.classifiers.core.driftdetection.ADWIN;
import moa.core.Measurement;
import moa.options.ClassOption;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.stream.IntStream;

import moa.classifiers.Classifier;
import org.openjdk.jol.info.GraphLayout;
import org.openjdk.jol.vm.VM;

import static moa.core.Measurement.getMeasurementNamed;

class TreeCommittee implements Serializable{

    private static final long serialVersionUID = 8961897277670201943L;
    protected Classifier[] treesCommittee;

    public TreeCommittee(Classifier baseLearner, int numTrees) {
        treesCommittee = new Classifier[numTrees];
        for (int i = 0; i < treesCommittee.length; i++) {
//            treesCommittee[i] = new StreamingGradientTree(featureInfo, options);
            treesCommittee[i] = baseLearner.copy();
        }
    }

    public HashMap getNumNodes() {
        int result = 0;
        HashMap<Integer,Object> treeInformation=new HashMap<Integer,Object>();//Creating HashMap
        for(int i = 0; i < treesCommittee.length; i++) {
            treeInformation.put(i, ((HoeffdingTree)treesCommittee[i]).activeLeafNodeCount);
//            result += treesCommittee[i].getNumNodes();
        }

        return treeInformation;
    }
//
//    public int getNumNodeUpdates() {
//        int result = 0;
//
//        for(int i = 0; i < treesCommittee.length; i++) {
//            result += treesCommittee[i].getNumNodeUpdates();
//        }
//
//        return result;
//    }
//
//    public int getNumSplits() {
//        int result = 0;
//
//        for(int i = 0; i < treesCommittee.length; i++) {
//            result += treesCommittee[i].getNumSplits();
//        }
//
//        return result;
//    }
//
//    public int getMaxDepth() {
//        int result = 0;
//
//        for(int i = 0; i < treesCommittee.length; i++) {
//            result = Math.max(treesCommittee[i].getDepth(), result);
//        }
//
//        return result;
//    }

//    public int getNumTrees() {
//        return treesCommittee.length;
//    }

    //    public void randomlyInitialize(Random rng, double predBound) {
//        for(StreamingGradientTree t : treesCommittee) {
//            t.randomlyInitialize(rng, predBound);
//        }
//    }

    static void modelUpdate(Classifier c, Instance inst, int multipleIterationByHessianCeiling, double hessian){
        double trainTimes = multipleIterationByHessianCeiling > 1 ? Math.ceil(hessian * multipleIterationByHessianCeiling) : 1.0;
        for (int i=0; i < (int) trainTimes; i++){
            c.trainOnInstance(inst);
        }
    }

    public void update(Instance[] instArray, int multipleIterationByHessianCeiling, GradHess[] gradHess) {
        if (treesCommittee.length == 1){
            IntStream.range(0, treesCommittee.length)
                    .forEach(i -> modelUpdate(treesCommittee[i],instArray[i], multipleIterationByHessianCeiling, gradHess[i].hessian ));
        }else{
            IntStream.range(0, treesCommittee.length)
                    .parallel()
                    .forEach(i -> modelUpdate(treesCommittee[i],instArray[i], multipleIterationByHessianCeiling, gradHess[i].hessian ));
        }

    }

    public void update(Instance inst) {
        if (treesCommittee.length == 1){
            IntStream.range(0, treesCommittee.length)
                    .forEach(i -> treesCommittee[i].trainOnInstance(inst));
        }else {
            IntStream.range(0, treesCommittee.length)
                    .parallel()
                    .forEach(i -> treesCommittee[i].trainOnInstance(inst));
        }
    }


    public double[] predict(Instance inst) {
        double[] v = new double[treesCommittee.length];
        if (treesCommittee.length == 1){
            IntStream.range(0, treesCommittee.length)
                    .forEach(i -> v[i] = treesCommittee[i].getVotesForInstance(inst)[0]);
        }else {
//            return IntStream.range(0, treesCommittee.length)
//                    .parallel()
//                    .mapToDouble(i -> treesCommittee[i].getVotesForInstance(inst)[0])
//                    .toArray();
            IntStream.range(0, treesCommittee.length)
                    .parallel()
                    .forEach(i -> v[i] = treesCommittee[i].getVotesForInstance(inst)[0]);
        }
        return v;
    }
}
public class BoostingTreePredictor extends BoostingCommittee implements Serializable, MultiClassClassifier, Regressor, SemiSupervisedLearner {

    private static final long serialVersionUID = 1L;

    protected TreeCommittee treeCommittee;

    protected int mInstances;

public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
        "Classifier to train on instances.", Classifier.class, "trees.HoeffdingRegressionTree -k -v -n HoeffdingNumericAttributeClassObserver -d HoeffdingNominalAttributeClassObserver -g 50 -c 0.05");

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
        treeCommittee = null;
        mInstances = 0;
    }

    @Override
    public long measureByteSize() {
        long b = 0;
        // get shallow size of this
        b = VM.current().sizeOf(this);
        if ((treeCommittee!= null) && (treeCommittee.treesCommittee != null)){

            long[] byteSize = new long[treeCommittee.treesCommittee.length];
            // get deep size of each item
            if (treeCommittee.treesCommittee.length == 1) {
                IntStream.range(0, treeCommittee.treesCommittee.length)
                        .forEach(i -> byteSize[i] = treeCommittee.treesCommittee[i].measureByteSize());
//                        .forEach(i -> byteSize[i] = GraphLayout.parseInstance(treeCommittee.treesCommittee[i]).totalSize());
            }else{
                IntStream.range(0, treeCommittee.treesCommittee.length)
                        .parallel()
                        .forEach(i -> byteSize[i] = treeCommittee.treesCommittee[i].measureByteSize());
//                        .forEach(i -> byteSize[i] = GraphLayout.parseInstance(treeCommittee.treesCommittee[i]).totalSize());
            }
            for (int i = 0; i < treeCommittee.treesCommittee.length; i++) {
                b += byteSize[i];
            }
        }
        return b;
    }

    protected TreeCommittee createTrees(Classifier baseLearner, int numOutputs) {
        return new TreeCommittee(baseLearner, numOutputs);
    }

    public void trainOnInstanceImpl(Instance inst, GradHess[] gradHess, double[] raw /* only for semiSupervisedOption */) {
        String currentMethod = new Exception().getStackTrace()[0].getMethodName();
        throw new UnsupportedOperationException(this.getClass().getName() + " " +currentMethod);
    }

    @Override
    public ArrayList<ArrayList<HashMap<String,String>>> getCommitteeInfo() {
        if (treeCommittee == null){
            return null;
        }
        ArrayList<ArrayList<HashMap<String,String>>> committeeInformation=new ArrayList<ArrayList<HashMap<String,String>>>();//Creating HashMap
        for(int i = 0; i < treeCommittee.treesCommittee.length; i++) {
//            ArrayList<String> treeInformation = new ArrayList<>();
            ArrayList<HashMap<String,String>> treeInformation1 = new ArrayList<HashMap<String,String>>();
            HashMap<String,String> hashMap=new HashMap<String,String>();
            long n = 0;
            long splitsByConfidence = 0;
            long splitsByHBound = 0;
            long splitsByHBoundSmallerThanTieThreshold = 0;
            long totalSplits = 0;
            String treeType = "U:";
            if (treeCommittee.treesCommittee[i] instanceof HoeffdingTree){
                n = ((HoeffdingTree)treeCommittee.treesCommittee[i]).activeLeafNodeCount;
                treeType = "H:";
            }else if (treeCommittee.treesCommittee[i] instanceof FIMTDD){
                n = ((FIMTDD)treeCommittee.treesCommittee[i]).leafNodeCount;
                splitsByConfidence = ((FIMTDD)treeCommittee.treesCommittee[i]).splitsByConfidence;
                splitsByHBound = ((FIMTDD)treeCommittee.treesCommittee[i]).splitsByHBound;
                splitsByHBoundSmallerThanTieThreshold = ((FIMTDD)treeCommittee.treesCommittee[i]).splitsByHBoundSmallerThanTieThreshold;
                totalSplits = ((FIMTDD)treeCommittee.treesCommittee[i]).splitNodeCount;
                treeType = "F:";
            } else if (treeCommittee.treesCommittee[i] instanceof StreamingGradientTreePredictor){
                n = ((StreamingGradientTreePredictor)treeCommittee.treesCommittee[i]).mTrees != null ?
                        ((StreamingGradientTreePredictor)treeCommittee.treesCommittee[i]).mTrees.getNumNodes(): 0;
                treeType = "S:";
            }
//            String s =".";
//            String sizeString = treeType;
//            sizeString += new String(new char[n]).replace("\0", s); // put n s into sizeString
//            treeInformation.add(sizeString);
            hashMap.put("type", treeType);
            hashMap.put("numNodes", "" + n);
            hashMap.put("splitsByConfidence", "" + splitsByConfidence);
            hashMap.put("splitsByHBound", "" + splitsByHBound);
            hashMap.put("splitsByHBoundSmallerThanTieThreshold", "" + splitsByHBoundSmallerThanTieThreshold);
            hashMap.put("totalSplits", "" + totalSplits);
            treeInformation1.add(hashMap);
//            committeeInformation.put(i,treeInformation);
            committeeInformation.add(treeInformation1);
        }

        return committeeInformation;
    }

    public void trainOnInstanceImpl(Instance[] instances, int multipleIterationByHessianCeiling, GradHess[] gradHess){
        mInstances++;

        if(treeCommittee == null) {
            Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
            treeCommittee = createTrees(baseLearner, committeeSize);
        }
        treeCommittee.update(instances, multipleIterationByHessianCeiling, gradHess);
    }

    public void trainOnInstanceImpl(Instance inst) {
        mInstances++;

        if(treeCommittee == null) {
            Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
            treeCommittee = createTrees(baseLearner, committeeSize);
        }
        treeCommittee.update(inst);
    }

    public double[] getScoresForInstance(Instance inst) {
        if(treeCommittee == null) {
            return getScoresWhenNullTree(committeeSize);
        }

        return treeCommittee.predict(inst);
    }

    public double[] getVotesForInstance(Instance inst) {
        String currentMethod = new Exception().getStackTrace()[0].getMethodName();
        throw new UnsupportedOperationException(this.getClass().getName() + " " +currentMethod);
//        return null;
    }

    @Override
    public void getModelDescription(StringBuilder in, int indent) {
    }

    static Measurement[] getTreeInfo(Classifier c){
        long n = 0;
        long splitsByConfidence = 0;
        long splitsByHBound = 0;
        long splitsByHBoundSmallerThanTieThreshold = 0;
        long totalSplits = 0;
        char treeType = 'U';
        if (c instanceof HoeffdingTree){
            n = ((HoeffdingTree)c).activeLeafNodeCount;
            treeType = 'H';
        }else if (c instanceof FIMTDD){
            n = ((FIMTDD)c).leafNodeCount;
            splitsByConfidence = ((FIMTDD)c).splitsByConfidence;
            splitsByHBound = ((FIMTDD)c).splitsByHBound;
            splitsByHBoundSmallerThanTieThreshold = ((FIMTDD)c).splitsByHBoundSmallerThanTieThreshold;
            totalSplits = ((FIMTDD)c).splitNodeCount;
            treeType = 'F';
        } else if (c instanceof StreamingGradientTreePredictor){
            n = ((StreamingGradientTreePredictor)c).mTrees != null ?
                    ((StreamingGradientTreePredictor)c).mTrees.getNumNodes(): 0;
            treeType = 'S';
        }

        return new Measurement[]{
                new Measurement("treeType", treeType),
                new Measurement("numNodes", n),
                new Measurement("splitsByConfidence", splitsByConfidence),
                new Measurement("splitsByHBound", splitsByHBound),
                new Measurement("splitsByHBoundSmallerThanTieThreshold", splitsByHBoundSmallerThanTieThreshold),
                new Measurement("totalSplits", totalSplits)
        };
    }
    @Override
    public Measurement[] getModelMeasurementsImpl() {
        double avgNumNodes = 0.0;
        double avgSplitsByConfidence = 0.0;
        double avgSplitsByHBound = 0.0;
        double avgSplitsByHBoundSmallerThanTieThreshold = 0.0;
        double avgTotalSplits = 0.0;
        if (treeCommittee.treesCommittee != null) {
            Measurement[][] m = new Measurement[treeCommittee.treesCommittee.length][];
            IntStream.range(0, treeCommittee.treesCommittee.length)
                    .parallel()
                    .forEach(i -> m[i] = getTreeInfo(treeCommittee.treesCommittee[i]));

            for (int i = 0; i < treeCommittee.treesCommittee.length; i++) {
                avgNumNodes += getMeasurementNamed("numNodes", m[i]).getValue();
                avgSplitsByConfidence += getMeasurementNamed("splitsByConfidence", m[i]).getValue();
                avgSplitsByHBound += getMeasurementNamed("splitsByHBound", m[i]).getValue();
                avgSplitsByHBoundSmallerThanTieThreshold += getMeasurementNamed("splitsByHBoundSmallerThanTieThreshold", m[i]).getValue();
                avgTotalSplits += getMeasurementNamed("totalSplits", m[i]).getValue();
            }

            avgNumNodes /= (1.0 * treeCommittee.treesCommittee.length);
            avgSplitsByConfidence /= (1.0 * treeCommittee.treesCommittee.length);
            avgSplitsByHBound /= (1.0 * treeCommittee.treesCommittee.length);
            avgSplitsByHBoundSmallerThanTieThreshold /= (1.0 * treeCommittee.treesCommittee.length);
            avgTotalSplits /= (1.0 * treeCommittee.treesCommittee.length);
        }

        return new Measurement[]{
                new Measurement("avgNumNodes", avgNumNodes),
                new Measurement("avgSplitsByConfidence", avgSplitsByConfidence),
                new Measurement("avgSplitsByHBound", avgSplitsByHBound),
                new Measurement("avgSplitsByHBoundSmallerThanTieThreshold", avgSplitsByHBoundSmallerThanTieThreshold),
                new Measurement("avgTotalSplits", avgTotalSplits)
        };
    }

    @Override
    public ImmutableCapabilities defineImmutableCapabilities() {
        if (this.getClass() == BoostingTreePredictor.class)
            return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
        else
            return new ImmutableCapabilities(Capability.VIEW_STANDARD);
    }
}
