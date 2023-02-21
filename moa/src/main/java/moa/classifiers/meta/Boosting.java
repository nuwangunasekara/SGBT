/*
 *    WeightedMajorityAlgorithm.java
 *    Copyright (C) 2007 University of Waikato, Hamilton, New Zealand
 *    @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */
package moa.classifiers.meta;

import com.github.javacliparser.*;
import com.henrygouk.sgt.GradHess;
import com.henrygouk.sgt.Objective;
import com.henrygouk.sgt.SoftmaxCrossEntropy;
import com.henrygouk.sgt.SquaredError;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.*;
import moa.classifiers.core.driftdetection.ADWIN;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.core.Utils;
import moa.options.ClassOption;
import moa.tasks.TaskMonitor;
import org.openjdk.jol.vm.VM;


import java.util.ArrayList;
import java.util.stream.IntStream;

import static moa.core.Measurement.getMeasurementNamed;

public class Boosting extends AbstractClassifier implements MultiClassClassifier {

    private static final long serialVersionUID = 1L;
    
    @Override
    public String getPurposeString() {
        return "Boosting algorithm for data streams.";
    }

    public FlagOption useSquaredLoss = new FlagOption("useSquaredLoss", 'K', "use Squared Loss");

    public FlagOption computeNegativeResidual = new FlagOption("computeNegativeResidual", 'N', "compute negative residual");

    public FlagOption clipPredictions = new FlagOption("clipPredictions", 'C', "clip predictions");
    public FlagOption useGradientOverHessianLabels = new FlagOption("useGradientOverHessianLabels", 'w', "use gradient/hessian labels and hessian as weights for instances");

    public FlagOption useCeilingForWeights = new FlagOption("useCeilingForWeights", 'c', "use ceiling function for weights (returns integer)");
    public FlagOption multiplyHessianBy10ForCeiling = new FlagOption("multiplyHessianBy10ForCeiling", 't', "multiply Hessian by 10 for Ceiling");

//    public FlagOption multipleIterationByHessian = new FlagOption("multipleIterationByHessian", 'M', "Multiple iteration by Hessian.");
    public IntOption multipleIterationByHessian = new IntOption("multiplyHForMultipleIterationByHC", 'M',
            "Multiply Hessian for Multiple iteration by Hessian Ceilling.", 1, 1, 100);

    public FlagOption useWeightOf1 = new FlagOption("useWeightOf1", 'W', "Use weight of 1.0 (do not use hessian as weights for instances). Used with -w");

    public FlagOption skipOnLossLessThan3SD = new FlagOption("skipOnLoss3SD", 'k', "skip on loss < 3 SD");

    public FlagOption useHeterogeneousEnsemble = new FlagOption("heterogeneousEnsemble", 'H', "Heterogeneous ensemble");
    public FlagOption useOneHotEncoding = new FlagOption("useOneHotEncoding", 'h', "useOneHotEncoding");

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train on instances.", Classifier.class, "trees.StreamingGradientTreePredictor");

    public IntOption numberOfboostingIterations = new IntOption("numberOfboostingIterations", 's',
            "The number of boosting iterations.", 10, 1, Integer.MAX_VALUE);
    public FloatOption learningRateOption = new FloatOption(
            "learningRate", 'L', "Learning rate",
            1.0, 0, 1.00);

    public IntOption subspaceSizeOption = new IntOption("subspaceSize", 'm',
            "# attributes per subset for each classifier. Negative values = totalAttributes - #attributes", 100, Integer.MIN_VALUE, Integer.MAX_VALUE);

    public IntOption skipTrainingRoughly = new IntOption("skipTrainingRoughly", 'S',
            "skip training roughly (specified # instances) - 1. Specified value needs to be > 1, for skipp training to happen.", 1, 1, Integer.MAX_VALUE);

//    public FlagOption resetEnsemble = new FlagOption("resetEnsemble", 'r', "Reset ensemble");

//    public FlagOption partialReset = new FlagOption("partialReset", 'p', "partial reset");
//    public IntOption skipResetAfterDrift = new IntOption("skipResetAfterDrift", 'R',
//            "skip reset after drift for R instances", 1, 1, Integer.MAX_VALUE);

//    public ClassOption driftDetectionMethodOption = new ClassOption("driftDetectionMethod", 'x',
//            "Change detector for drifts and its parameters", ChangeDetector.class, "ADWINChangeDetector -a 1.0E-6");


//    public IntOption driftGap = new IntOption("driftGap", 'G',
//            "Gap between two start of drifts", 0, 0, Integer.MAX_VALUE);
//    public FlagOption dynamicLearningRate = new FlagOption("dynamicLearningRate", 'D', "Dynamic Learning Rate");

//    public IntOption randomSeedOption = new IntOption("randomSeedOption",
//            'r', "randomSeedOption",
//            1,Integer.MIN_VALUE, Integer.MAX_VALUE);

    public ClassOption [] heterogeneousEnsembleClasses = {
            new ClassOption("baseLearner", 'l', "Classifier to train on instances.", Classifier.class,
//                    "trees.BoostingTreePredictor -l (trees.StreamingGradientTreePredictor -D 0.05 -L 0.05 -Y 0.5 -G 200)"),
                    "trees.BoostingTreePredictor -l (drift.DriftDetectionMethodClassifier -l (trees.StreamingGradientTreePredictor -D 0.05 -L 0.05 -Y 0.5 -W 400) -d (DDM -n 250 -o 2.5))"),
            new ClassOption("baseLearner", 'l', "Classifier to train on instances.", Classifier.class,
                    "trees.BoostingTreePredictor -l (trees.FIMTDD -s VarianceReductionSplitCriterion -g 25 -c 0.05 -e -p)")
    };

    public ListOption baselearnersOption = new ListOption("baseClassifiers", 'b',
            "The classifiers the heterogeneous ensemble consists of.",
            new ClassOption("learner", ' ', "", Classifier.class,
                    "trees.BoostingTreePredictor -l (trees.FIMTDD -s VarianceReductionSplitCriterion -g 25 -c 0.05 -e -p)"),
            new Option[] {
                    new ClassOption("FIMTDD_g50_c10", ' ', "", Classifier.class, "trees.BoostingTreePredictor -l (trees.FIMTDD -s VarianceReductionSplitCriterion -g 50 -c 0.1 -e -p)"),
                    new ClassOption("FIMTDD_g50_c05", ' ', "", Classifier.class, "trees.BoostingTreePredictor -l (trees.FIMTDD -s VarianceReductionSplitCriterion -g 50 -c 0.05 -e -p)"),
                    new ClassOption("FIMTDD_g25_c10", ' ', "", Classifier.class, "trees.BoostingTreePredictor -l (trees.FIMTDD -s VarianceReductionSplitCriterion -g 25 -c 0.1 -e -p)"),
                    new ClassOption("FIMTDD_g25_c05", ' ', "", Classifier.class, "trees.BoostingTreePredictor -l (trees.FIMTDD -s VarianceReductionSplitCriterion -g 25 -c 0.05 -e -p)"),
                    new ClassOption("SGT_DD", ' ', "", Classifier.class, "trees.BoostingTreePredictor -l (drift.DriftDetectionMethodClassifier -l (trees.StreamingGradientTreePredictor -D 0.05 -L 0.05 -Y 0.5 -W 400) -d (DDM -n 250 -o 2.5))")
            },
            ',');

    protected ArrayList<BoostingCommittee> baseLearnerArray;

    protected ArrayList<BoostingCommittee> booster;

    protected ADWIN lossEstimator;
    protected double skipCount = 0.0;

    protected BoostingCommittee baseLearner;
    private int committeeSize;

    protected ArrayList<ArrayList<Integer>> subspaces;
    protected ArrayList<ArrayList<Integer>> subSpacesForEachBoostingIteration;
    protected Objective mObjective;

    protected double[] lastPrediction = null;
//    protected ChangeDetector driftDetectorForBooster = null;
//    protected ArrayList<ChangeDetector> driftDetectorForEachEnsemble;
    private long instancesSeenAtTrainSinceReset;
    private long instancesSeenAtTrain;


    int [] baseLearnerIndex = null;
    @Override
    public void resetLearningImpl() {
        System.out.println("Re-setting booster.");
        if(this.booster != null){
            while (this.booster.size() > 0) {
                this.booster.remove(0);
            }
            this.booster = null;
        }

        lossEstimator = null;

        if(this.subSpacesForEachBoostingIteration != null){
            while (this.subSpacesForEachBoostingIteration.size() > 0) {
                this.subSpacesForEachBoostingIteration.remove(0);
            }
            this.subSpacesForEachBoostingIteration = null;
        }
//        if(this.driftDetectorForEachEnsemble != null){
//            while (this.driftDetectorForEachEnsemble.size() > 0) {
//                this.driftDetectorForEachEnsemble.remove(0);
//            }
//        }
//        this.driftDetectorForEachEnsemble = null;
        this.mObjective = null;
//        this.driftDetectorForBooster = null;
        this.instancesSeenAtTrainSinceReset = 0;
    }
    @Override
    public void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {

        monitor.setCurrentActivity("Materializing learner " + (0) + "...", -1.0);

        super.randomSeedOption.setValue(this.randomSeedOption.getValue());
        System.out.println("RANDOM seed: "+ super.randomSeedOption.getValue());
//        super.randomSeedOption.setValue(this.randomSeedOption.getValue());

        super.prepareForUseImpl(monitor, repository);

        baseLearner = (BoostingCommittee) baseLearnerOption.materializeObject(monitor, repository);
        baseLearnerArray = new ArrayList<>();
        Option[] learnerOptions = this.baselearnersOption.getList();
        for (int i = 0; i < learnerOptions.length; i++) {
//        for (int i = 0; i < heterogeneousEnsembleClasses.length; i++){
            baseLearnerArray.add((BoostingCommittee)((ClassOption) learnerOptions[i]).materializeObject(monitor, repository));
//            baseLearnerArray.add((BoostingCommittee)heterogeneousEnsembleClasses[i].materializeObject(monitor, repository));
        }

        if (monitor.taskShouldAbort()) {
            return;
        }

        monitor.setCurrentActivity("Preparing learner " + (0) + "...", -1.0);

        baseLearner.prepareForUse(monitor, repository);
        for (int i=0; i < baseLearnerArray.size(); i++){
            baseLearnerArray.get(i).prepareForUse(monitor, repository);
        }
    }

//    public static void setFeatureValuesArray(Instance inst, double[] featureValuesArrayToSet, boolean useOneHotEncoding){
//        int totalOneHotEncodedSize = 0;
//        int totalOneHotEncodedInstances = 0;
//        for(int i=0; i < inst.numInputAttributes(); i++){
//            int index = i + totalOneHotEncodedSize - totalOneHotEncodedInstances;
//            if (useOneHotEncoding && inst.attribute(i).isNominal() && (inst.attribute(i).numValues() > 2) ){
//                // Do one hot-encoding
//                featureValuesArrayToSet[index + (int)inst.value(i)] = 1.0f;
//                totalOneHotEncodedSize += inst.attribute(i).numValues();
//                totalOneHotEncodedInstances ++;
//            }else
//            {
//                featureValuesArrayToSet[index] = inst.value(i);
//            }
//        }
//    }
//    public static int getFeatureValuesArraySize(Instance inst, boolean useOneHotEncoding){
//        int totalOneHotEncodedSize = 0;
//        int totalOneHotEncodedInstances = 0;
//        for(int i=0; i < inst.numInputAttributes(); i++){
//            if (useOneHotEncoding && inst.attribute(i).isNominal() && (inst.attribute(i).numValues() > 2) ){
//                totalOneHotEncodedSize += inst.attribute(i).numValues();
//                totalOneHotEncodedInstances ++;
//            }
//        }
//        return inst.numInputAttributes() + totalOneHotEncodedSize - totalOneHotEncodedInstances;
//    }

    public static Instance getSubInstance(Instance instance, double weight, ArrayList<Integer> subSpaceFeaturesIndexes, boolean setLabel, double labelValue, boolean useOneHotEncoding){
        Instances subset;
        ArrayList<Attribute> attSub = new ArrayList<>();
        ArrayList<Double> v = new ArrayList<>();
        Attribute classAttribute;
        int totalOneHotEncodedSize = 0;
        int totalOneHotEncodedInstances = 0;
        int i = 0;
        // Add attributes of the selected subset
        for (Integer featuresIndex : subSpaceFeaturesIndexes) {
//            attSub.add(instance.attribute(featuresIndex));
            int index = i + totalOneHotEncodedSize - totalOneHotEncodedInstances;
            if (useOneHotEncoding && instance.attribute(featuresIndex).isNominal() && (instance.attribute(featuresIndex).numValues() > 2) ){
                // Do one hot-encoding
//                featureValuesArrayToSet[index + (int)instance.value(i)] = 1.0f;
                for (int j = 0; j < instance.attribute(featuresIndex).numValues(); j++){
//                    attSub.set(index + j, new Attribute(""));
//                    v.set(index + j, Double.valueOf(0.0));
                    attSub.add(new Attribute(""));
                    v.add(Double.valueOf(0.0));
                }
                v.set(index + (int)instance.value(featuresIndex), Double.valueOf(1.0));

                totalOneHotEncodedSize += instance.attribute(featuresIndex).numValues();
                totalOneHotEncodedInstances ++;
//                attSub.set(index + (int)instance.value(i), new Attribute(""));
            }else {
//                featureValuesArrayToSet[index] = inst.value(i);
//                attSub.set(index, instance.attribute(featuresIndex));
//                v.set(index, Double.valueOf(instance.value(subSpaceFeaturesIndexes.get(featuresIndex))));
                attSub.add(instance.attribute(featuresIndex));
                v.add(Double.valueOf(instance.value(featuresIndex)));
            }
            i++;
        }
        // add class attribute
        if (setLabel){
            // adds a numeric class attribute
            classAttribute = new Attribute("classAttribute");
        }else{
            classAttribute = instance.classAttribute();
        }
        attSub.add(classAttribute);
        v.add(Double.valueOf(setLabel ? labelValue : instance.classValue()));
        subset = new Instances("Subsets Candidate Instances", attSub, 100);
        subset.setClassIndex(subset.numAttributes()-1);

//        prepareRandomSubspaceInstance(instance,1);
        // If there is any instance lingering in the subset, remove it.
//        while(subset.numInstances() > 0)
//            subset.delete(0);

//        double[] values = new double[subset.numAttributes()];
//        for(int j = 0 ; j < subset.numAttributes() -1; ++j) {
//            values[j] = instance.value(subSpaceFeaturesIndexes.get(j));
//        }

        // Set the class value for each value array.
//        values[values.length-1] = setLabel ? labelValue : instance.classValue();
        double[] values = new double[v.size()];
        for (int k=0; k < v.size(); k++) {
            values[k] = v.get(k).doubleValue();
        }
//        double[] values = v.stream()
//                .mapToDouble(Double::doubleValue).toArray();
        DenseInstance subInstance = new DenseInstance(weight, values);
        subInstance.setWeight(weight);
        subInstance.setDataset(subset);

        subset.add(subInstance);
        return subInstance;
    }

    public void initEnsemble(Instance inst){
        System.out.println("Initializing booster.");
        Attribute target = inst.classAttribute();

        if (useSquaredLoss.isSet()){
            System.out.println("Using SquaredError");
            mObjective = new SquaredError();
        }else{
            System.out.println("Using SoftmaxCrossEntropy");
            mObjective = new SoftmaxCrossEntropy();
        }


        if (booster == null){
            booster = new ArrayList<>();
        } else {
            while(booster.size() > 0)
                booster.remove(0);
        }
//        if (driftDetectorForEachEnsemble == null){
//            driftDetectorForEachEnsemble = new ArrayList<>();
//        } else {
//            while(driftDetectorForEachEnsemble.size() > 0)
//                driftDetectorForEachEnsemble.remove(0);
//        }

        if(target.isNominal()) {
            if (useSquaredLoss.isSet()){
                committeeSize = target.numValues() - 1;
            }else {
                committeeSize = target.numValues() - 1;
            }
            baseLearner.committeeSize = committeeSize;
            for (int i=0; i < this.baseLearnerArray.size(); i++){
                this.baseLearnerArray.get(i).committeeSize = committeeSize;
            }
        } else {
            String currentMethod = new Exception().getStackTrace()[0].getMethodName();
            throw new UnsupportedOperationException(this.getClass().getName() + " " +currentMethod+ " " + "Regression/Numeric targets are not supported yet");
        }
        System.out.println("CommitteeSize: " + committeeSize);

        if (lossEstimator != null){
            lossEstimator = null;
        }
        lossEstimator = new ADWIN(1.0E-3);

        baseLearnerIndex = this.classifierRandom.ints(0, baseLearnerArray.size()).limit(numberOfboostingIterations.getValue()).toArray();
        for (int i = 0; i< numberOfboostingIterations.getValue(); i ++) {
            if (useGradientOverHessianLabels.isSet() && useHeterogeneousEnsemble.isSet()){
                booster.add((BoostingCommittee) baseLearnerArray.get(baseLearnerIndex[i]).copy());
            }else {
                booster.add((BoostingCommittee) baseLearner.copy());
            }
        }

//        for (int i=0; i< ensembleSize.getValue(); i ++) {
////            this.driftDetectorForEachEnsemble.add(((ChangeDetector) getPreparedClassOption(this.driftDetectionMethodOption)).copy());
//        }

//        this.driftDetectorForBooster = ((ChangeDetector) getPreparedClassOption(this.driftDetectionMethodOption)).copy();


        // #1 Select the size of k, it depends on 2 parameters (subspaceSizeOption and subspaceModeOption).
        int k = this.subspaceSizeOption.getValue();

        int n = inst.numAttributes()-1; // Ignore the class label by subtracting 1

        double percent = k < 0 ? (100 + k)/100.0 : k / 100.0;
        k = (int) Math.round(n * percent);

        if(Math.round(n * percent) < 2)
            k = (int) Math.round(n * percent) + 1;

        // k is negative, use size(features) + -k
        if(k < 0)
            k = n + k;

        // #2 generate the subspaces

        if(k != 0 && k < n) {
            // For low dimensionality it is better to avoid more than 1 classifier with the same subspaces,
            // thus we generate all possible combinations of subsets of features and select without replacement.
            // n is the total number of features and k is the actual size of the subspaces.
            if(n <= 20 || k < 2) {
                if(k == 1 && inst.numAttributes() > 2)
                    k = 2;
                // Generate all possible combinations of size k
                this.subspaces = StreamingRandomPatches.allKCombinations(k, n);
                for(int i = 0; this.subspaces.size() < this.numberOfboostingIterations.getValue() ; ++i) {
                    i = i == this.subspaces.size() ? 0 : i;
                    ArrayList<Integer> copiedSubspace = new ArrayList<>(this.subspaces.get(i));
                    this.subspaces.add(copiedSubspace);
                }
            }
            // For high dimensionality we can't generate all combinations as it is too expensive (memory).
            // On top of that, the chance of repeating a subspace is lower, so we can just randomly generate
            // subspaces without worrying about repetitions.
            else {
                this.subspaces = StreamingRandomPatches.localRandomKCombinations(k, n,
                        this.numberOfboostingIterations.getValue(), this.classifierRandom);
            }
        }else if (k == n){
            this.subspaces = StreamingRandomPatches.localRandomKCombinations(k, n,
                    this.numberOfboostingIterations.getValue(), this.classifierRandom);
        }

        int[] subSpaceIndexes = this.classifierRandom.ints(0, subspaces.size()).distinct().limit(numberOfboostingIterations.getValue()).toArray();
        subSpacesForEachBoostingIteration = new ArrayList<>();
        for (int i = 0; i < numberOfboostingIterations.getValue(); i++){
            subSpacesForEachBoostingIteration.add(this.subspaces.get(subSpaceIndexes[i]));
        }
        System.out.println("Ensemble size: "+ booster.size() + " subSpacesForEnsemble size:" + subSpacesForEachBoostingIteration.size());
    }

//    public void reInitEnsembleFrom(int from){
//        System.out.println("Re-init Ensemble from " + from);
//        if (ensemble == null){
//            ensemble = new ArrayList<>();
//        } else {
//            for (int i = ensembleSize.getValue()-1; i >= from; i--){
//                ensemble.remove(i);
//            }
//        }
//        for (int i = from; i < ensembleSize.getValue(); i++) {
//            if (useWeightedInstances.isSet() && useHeterogeneousEnsemble.isSet()){
//                ensemble.add((BoostingCommittee) baseLearnerArray.get(baseLearnerIndex[i]).copy());
//            }else {
//                ensemble.add((BoostingCommittee) baseLearner.copy());
//            }
//            driftDetectorForEachEnsemble.get(i).resetLearning();
//        }
//        instancesSeenAtTrainSinceReset = 0;
//    }

//    boolean detectChange(double input, ChangeDetector driftDetector){
//        boolean upwardDriftDetected = false;
//        double previousLossEstimation = driftDetector.getEstimation();
//        driftDetector.input(input);
//        double currentLossEstimation = driftDetector.getEstimation();
//        if (driftDetector.getChange()){
//            System.out.println("Drift detected. at " + instancesSeenAtTrain);
//            if (currentLossEstimation > previousLossEstimation){
//                System.out.println("At " + instancesSeenAtTrain + " since reset " +instancesSeenAtTrainSinceReset + ": currentLossEstimation: "+ currentLossEstimation + " > previousLossEstimation: " + previousLossEstimation);
//                if (instancesSeenAtTrainSinceReset > skipResetAfterDrift.getValue()) {
//                    upwardDriftDetected = true;
//                }
//            }
//        }
//        return upwardDriftDetected;
//    }

    public void trainBoosterUsingSoftmaxCrossEntropyLoss(Instance inst){
        instancesSeenAtTrain++;
        instancesSeenAtTrainSinceReset++;

//        if ((ensemble == null) || ((driftGap.getValue() > 0) && (instancesSeenAtTrain % driftGap.getValue() == 0))) {
//            initEnsemble(inst);
//        }
//        double loss = 0.0;
        double[] groundTruth = new double[inst.numClasses()];
        groundTruth[(int) inst.classValue()] = 1.0;

//        loss = (new SoftmaxCrossEntropy()).loss(groundTruth, lastPrediction); // assumes test then train
//        if (detectChange(loss, driftDetectorForBooster)){
//            if (resetEnsemble.isSet()){
//                this.resetLearningImpl();
//            }
//        }
//        if ((driftGap.getValue() > 0) && (instancesSeenAtTrain % driftGap.getValue() == 0)){
//            this.resetLearningImpl();
//        }
//        loss = 0.0;
        if (booster == null) {
            initEnsemble(inst);
        }
        // get initial score, this is 0.0 for all the trees in the committee
        DoubleVector rawScore = new DoubleVector(BoostingCommittee.getScoresWhenNullTree(committeeSize));
        double loss = 0.0;
        for (int m = 0; m < booster.size(); m++) {
            Instance subInstance;
            // compute Derivatives (g and h) using y and summed up raw score, for all the trees in the committee
            // computeNegativeResidual=true only when NOT useWeightedInstances.isSet()
            // clipPredictions=true only when NOT useWeightedInstances.isSet()
//            GradHess[] gradHess = mObjective.computeDerivatives(groundTruth, rawScore.getArrayRef(), !useWeightedInstances.isSet());

            // at m th iteration, gets the adjustment by the m th committee considering all the previous adjustments
            GradHess[] gradHess = mObjective.computeDerivatives(groundTruth, rawScore.getArrayRef(), computeNegativeResidual.isSet(), clipPredictions.isSet());
            loss += mObjective.lossForAllClasses;
            boolean skipTrain = false;
            if (skipOnLossLessThan3SD.isSet() && (m == 0)){
                double sd = Math.sqrt(lossEstimator.getVariance());
                if (mObjective.lossForAllClasses < ( lossEstimator.getEstimation() - ( 3 * sd) ) ) {
                    System.out.println(mObjective.lossForAllClasses + "," + lossEstimator.getEstimation() + "," + sd);
                    skipTrain = true;
                }
            }
            if (skipTrain == true) {
                skipCount += 1.0;
                break;
            }

//            loss += mObjective.loss(groundTruth);
//            loss = (new SquaredError()).loss(groundTruth, rawScore.getArrayRef());
//            if (detectChange(loss, driftDetectorForEachEnsemble.get(m))){
//                if(partialReset.isSet()){
//                    reInitEnsembleFrom(m);
//                    driftDetectorForEachEnsemble.get(m).input(loss);
//                }
//            }
//            subInstance = inst;
            // create a sub instance from the inst
            boolean setLabel = false;
            if(useGradientOverHessianLabels.isSet()){
                setLabel = true;
            }
            subInstance = getSubInstance(inst, 1.0, subSpacesForEachBoostingIteration.get(m), setLabel, -1, useOneHotEncoding.isSet());

            if(useGradientOverHessianLabels.isSet()){
                //create sub instance for each committee member

                Instance[] subInstArray = new Instance[gradHess.length];
                if (gradHess.length == 1) {
                    subInstArray[0] = subInstance;
                }
                else{
                    IntStream.range(0, gradHess.length)
                            .forEach(i -> subInstArray[i] = subInstance.copy());
//                        .mapToObj(i -> subInstance.copy()).toArray(Instance[]::new);
                }
                if (!useWeightOf1.isSet()) { // set each sub instance weight to hessian, when doNotUseHessianAsWeight is NOT set
//                    useCeilingForWeights.isSet(), pass Math.ceil(gradHess[i].hessian) as weight
                    IntStream.range(0, subInstArray.length)
//                            .parallel()
                            .forEach(i -> subInstArray[i].setWeight( useCeilingForWeights.isSet() ? Math.ceil(multiplyHessianBy10ForCeiling.isSet() ? gradHess[i].hessian * 10 : gradHess[i].hessian) : gradHess[i].hessian));
                }
                // set each sub instance pseudo label to gradient/hessian
                IntStream.range(0, subInstArray.length)
//                        .parallel()
                        .forEach(i -> subInstArray[i].setClassValue(gradHess[i].gradient/gradHess[i].hessian));

//                double avgHessian = 0.0;
//                for (int i=0; i < subInstArray.length; i++){
//                    avgHessian += gradHess[i].hessian;
//                }
//                avgHessian = avgHessian / subInstArray.length;
//                double trainTimes = multipleIterationByHessian.isSet() ? Math.ceil(avgHessian * 10) : 1.0;
                // train each member of the committee using sub instance with relevant weight and pseudo-label
                booster.get(m).trainOnInstanceImpl(subInstArray, multipleIterationByHessian.getValue(), gradHess);
//                for (int i=0; i < (int) trainTimes; i++){
//                    booster.get(m).trainOnInstanceImpl(subInstArray, multipleIterationByHessian.isSet(), gradHess);
//                }
            }else { // use unweighted Instances
                // train using StreamingGradientTreePredictor, each member of the committee using sub instance
                booster.get(m).trainOnInstanceImpl(
                        subInstance,
                        gradHess,
                        null  /* Need to pass rawScore, only for SGT semiSupervisedOption. We don't use it here */);
            }

            // get the score from the committee for current subInstance (here we use subInstance for useWeightedInstances==true, as we do not need the label)
            DoubleVector currentScore = new DoubleVector(booster.get(m).getScoresForInstance(subInstance));
            // scale the score by learning rate
            double learningRate = learningRateOption.getValue();
//            if (dynamicLearningRate.isSet()) {
//                for (int i = 0; i < gradHess.length; i++) {
//                    learningRate += Math.abs(gradHess[i].gradient);
//                }
//                learningRate = learningRate / gradHess.length * learningRateOption.getValue();
//            }else{
//                learningRate = learningRateOption.getValue();
//            }
            currentScore.scaleValues(learningRate);
            // add the current sore to existing raw score
            rawScore.addValues(currentScore);
        }
        lossEstimator.setInput(loss);
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        if ((this.instancesSeenAtTrain % 1000) == 0){
            System.gc();
//            ClassLayout layout = ClassLayout.parseInstance(this);
//            System.out.println("Size of: " +ClassLayout.parseInstance(this).instanceSize());
        }
        if ((this.skipTrainingRoughly.getValue() > 1) && (this.classifierRandom.nextInt(this.skipTrainingRoughly.getValue()) == 0) ){
            // skip training
            return;
        }
        trainBoosterUsingSoftmaxCrossEntropyLoss(inst);
    }

    static double[] getScoreFromSubInstance(Instance inst, ArrayList<Integer> subSpaceFeaturesIndexes, boolean setLabel, BoostingCommittee b, boolean useOneHotEncoding){
        Instance subInstance = getSubInstance(inst, 1.0, subSpaceFeaturesIndexes, setLabel, -1, useOneHotEncoding);
        return b.getScoresForInstance(subInstance);
    }


    public DoubleVector getRawScoreForInstance(Instance inst) {
        final boolean setLabel = useGradientOverHessianLabels.isSet() ? true : false;
//        if(useGradientOverHessianLabels.isSet()){
//            setLabel = true;
//        }
        DoubleVector rawScore = new DoubleVector(BoostingCommittee.getScoresWhenNullTree(committeeSize));
//        Instance[] subInstanceArray = new Instance[booster.size()];
//        IntStream.range(0, booster.size())
//                .forEach(i -> subInstanceArray[i] = getSubInstance(inst, 1.0, subSpacesForEachBoostingIteration.get(0), setLabel, -1, useOneHotEncoding.isSet()));

//        for (int m = 0; m < booster.size(); m++) {
//            Instance subInstance = getSubInstance(inst, 1.0, subSpacesForEachBoostingIteration.get(m), setLabel, -1, useOneHotEncoding.isSet());
//            rawScore.addValues(booster.get(m).getScoresForInstance(subInstance));
//        }

        double s[][] = new double[booster.size()][];
        if (booster.size() == 1){
            s[0] = getScoreFromSubInstance(inst, subSpacesForEachBoostingIteration.get(0), setLabel, booster.get(0), useOneHotEncoding.isSet());
//            s[0] = booster.get(0).getScoresForInstance(subInstanceArray[0]);
        } else {
            IntStream.range(0, booster.size())
                    .parallel()
                    .forEach(m -> s[m] = getScoreFromSubInstance(inst, subSpacesForEachBoostingIteration.get(m), setLabel, booster.get(m), useOneHotEncoding.isSet()));

//            IntStream.range(0, booster.size())
//                    .parallel()
//                    .forEach(m -> s[m] = booster.get(m).getScoresForInstance(subInstanceArray[m]));
        }
        for (int i = 0; i < booster.size(); i++) {
            rawScore.addValues(s[i]);
        }
        return rawScore;
    }

    @Override
    public boolean correctlyClassifies(Instance inst) {
//        Assumes test then train evaluation set up
        return Utils.maxIndex(lastPrediction) == (int) inst.classValue();
    }

    public double[] getVotesForInstance(Instance inst) {
        if (booster == null) {
            initEnsemble(inst);
        }

        lastPrediction = mObjective.transfer(getRawScoreForInstance(inst).getArrayCopy());
        return lastPrediction;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        // TODO Auto-generated method stub
    }

    static Measurement[] getModelMeasurementsS (
            Classifier c){
        return c.getModelMeasurements(0.0);
    }
    @Override
    protected Measurement[] getModelMeasurementsImpl() {

        double avgNumNodes = 0.0;
        double avgSplitsByConfidence = 0.0;
        double avgSplitsByHBound = 0.0;
        double avgSplitsByHBoundSmallerThanTieThreshold = 0.0;
        double avgTotalSplits = 0.0;
        if (booster != null) {
            Measurement[][] m = new Measurement[booster.size()][];
            IntStream.range(0, booster.size())
                    .parallel()
                    .forEach(i -> m[i] = getModelMeasurementsS(booster.get(i)));

            for (int i = 0; i < booster.size(); i++) {
                avgNumNodes += getMeasurementNamed("avgNumNodes", m[i]).getValue();
                avgSplitsByConfidence += getMeasurementNamed("avgSplitsByConfidence", m[i]).getValue();
                avgSplitsByHBound += getMeasurementNamed("avgSplitsByHBound", m[i]).getValue();
                avgSplitsByHBoundSmallerThanTieThreshold += getMeasurementNamed("avgSplitsByHBoundSmallerThanTieThreshold", m[i]).getValue();
                avgTotalSplits += getMeasurementNamed("avgTotalSplits", m[i]).getValue();
            }

            avgNumNodes /= (1.0 * booster.size());
            avgSplitsByConfidence /= (1.0 * booster.size());
            avgSplitsByHBound /= (1.0 * booster.size());
            avgSplitsByHBoundSmallerThanTieThreshold /= (1.0 * booster.size());
            avgTotalSplits /= (1.0 * booster.size());
        }

        return new Measurement[]{
                new Measurement("avgNumNodes", avgNumNodes),
                new Measurement("avgSplitsByConfidence", avgSplitsByConfidence),
                new Measurement("avgSplitsByHBound", avgSplitsByHBound),
                new Measurement("avgSplitsByHBoundSmallerThanTieThreshold", avgSplitsByHBoundSmallerThanTieThreshold),
                new Measurement("avgTotalSplits", avgTotalSplits),
                new Measurement("skipCount", skipCount)
        };
    }

    @Override
    public long measureByteSize() {
        long b = 0;
        // get shallow size of this
        b = VM.current().sizeOf(this);
        if (booster != null) {
            long[] byteSize = new long[booster.size()];
            // get deep size of each item
            if (booster.size() == 1) {
                IntStream.range(0, booster.size())
                        .forEach(i -> byteSize[i] = booster.get(i).measureByteSize());
            }else{
                IntStream.range(0, booster.size())
                        .parallel()
                        .forEach(i -> byteSize[i] = booster.get(i).measureByteSize());
//                .forEach(i -> byteSize[i] = GraphLayout.parseInstance(booster.get(i)).totalSize());
            }
            for (int i = 0; i < booster.size(); i++) {
                b += byteSize[i];
            }
        }
        return b;
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

//    @Override
//    public Classifier[] getSubClassifiers() {
//        return this.ensemble.clone();
//    }

    @Override
    public ImmutableCapabilities defineImmutableCapabilities() {
        if (this.getClass() == Boosting.class)
            return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
        else
            return new ImmutableCapabilities(Capability.VIEW_STANDARD);
    }

}
