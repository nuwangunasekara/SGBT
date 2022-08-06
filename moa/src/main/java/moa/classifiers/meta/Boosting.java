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

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
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
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.classifiers.trees.StreamingGradientTreePredictor;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.options.ClassOption;
import moa.tasks.TaskMonitor;


import java.util.ArrayList;
import java.util.Random;

public class Boosting extends AbstractClassifier implements MultiClassClassifier {

    private static final long serialVersionUID = 1L;
    
    @Override
    public String getPurposeString() {
        return "Boosting algorithm for data streams.";
    }

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train on instances.", Classifier.class, "trees.StreamingGradientTreePredictor -G 200 -W 1");
//    trees.StreamingGradientTreePredictor -G 10 -W 1
//    trees.FIMTDD

    public IntOption ensembleSize = new IntOption("ensembleSize", 's',
            "The maximum number of classifiers in the ensemble.", 10, 1, Integer.MAX_VALUE);
    public FloatOption learningRateOption = new FloatOption(
            "learningRate", 'L', "Learning rate",
            1.0, 0, 1.00);

    public IntOption subspaceSizeOption = new IntOption("subspaceSize", 'm',
            "# attributes per subset for each classifier. Negative values = totalAttributes - #attributes", 100, Integer.MIN_VALUE, Integer.MAX_VALUE);

    protected ArrayList<StreamingGradientTreePredictor> ensemble;

    protected Classifier baseLearner;

    protected ArrayList<ArrayList<Integer>> subspaces;
    protected ArrayList<ArrayList<Integer>> subSpacesForEnsemble;
    protected Objective mObjective;

    @Override
    public void prepareForUseImpl(TaskMonitor monitor,
            ObjectRepository repository) {

        monitor.setCurrentActivity("Materializing learner " + (0)
                + "...", -1.0);
        this.baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);

        if (monitor.taskShouldAbort()) {
            return;
        }
        monitor.setCurrentActivity("Preparing learner " + (0) + "...",
                -1.0);
        this.baseLearner.prepareForUse(monitor, repository);

        if (monitor.taskShouldAbort()) {
            return;
        }
        super.prepareForUseImpl(monitor, repository);
    }

    @Override
    public void resetLearningImpl() {
//        for (int i = 0; i < this.ensemble.length; i++) {
//            this.ensemble[i].resetLearning();
//        }
    }
    public Instance getSubInstance(Instance instance, double weight, ArrayList<Integer> featuresIndexes){
        Instances subset;
        ArrayList<Attribute> attSub = new ArrayList<>();

        // Add attributes of the selected subset
        for (Integer featuresIndex : featuresIndexes) {
            attSub.add(instance.attribute(featuresIndex));
        }
        // add class attribute
        attSub.add(instance.classAttribute());
        subset = new Instances("Subsets Candidate Instances", attSub, 100);
        subset.setClassIndex(subset.numAttributes()-1);

//        prepareRandomSubspaceInstance(instance,1);
        // If there is any instance lingering in the subset, remove it.
        while(subset.numInstances() > 0)
            subset.delete(0);

        double[] values = new double[subset.numAttributes()];
        for(int j = 0 ; j < subset.numAttributes() -1; ++j) {
            values[j] = instance.value(featuresIndexes.get(j));
        }

        // Set the class value for each value array.
        values[values.length-1] = instance.classValue();
        DenseInstance subInstance = new DenseInstance(1.0, values);
        subInstance.setWeight(weight);
        subInstance.setDataset(subset);

        subset.add(subInstance);
        return subInstance;
    }

    public void initEnsemble(Instance inst){
        //re-init, could go into a new function
        Attribute target = inst.classAttribute();
        if(target.isNominal()) {
            mObjective = new SoftmaxCrossEntropy();
        } else {
            mObjective = new SquaredError();
        }

        if (ensemble == null){
            ensemble = new ArrayList<>();
        } else {
            while(ensemble.size() > 0)
                ensemble.remove(0);
        }

        for (int i=0; i< ensembleSize.getValue(); i ++) {
            ensemble.add((StreamingGradientTreePredictor) baseLearner.copy());
        }


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
                for(int i = 0 ; this.subspaces.size() < this.ensembleSize.getValue() ; ++i) {
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
                        this.ensembleSize.getValue(), this.classifierRandom);
            }
        }else if (k == n){
            this.subspaces = StreamingRandomPatches.localRandomKCombinations(k, n,
                    this.ensembleSize.getValue(), this.classifierRandom);
        }

        int[] subSpaceIndexes = new Random(this.randomSeed).ints(0, subspaces.size()).distinct().limit(ensembleSize.getValue()).toArray();
        subSpacesForEnsemble = new ArrayList<>();
        for (int i=0; i < ensembleSize.getValue(); i++){
            subSpacesForEnsemble.add(this.subspaces.get(subSpaceIndexes[i]));
        }
    }

    public void trainBoosterUsingSquareLoss(Instance inst, boolean trainF0){
        Instance m_1_Instance = getSubInstance(inst, 1, subSpacesForEnsemble.get(0));
        if (trainF0) {
            ensemble.get(0).trainOnInstance(m_1_Instance);
        }
        double yBAcu = 0.0;
        for (int m = 1; m < ensemble.size(); m++) {
            yBAcu += learningRateOption.getValue() * ensemble.get(m-1).getVotesForInstance(m_1_Instance)[0];
            double r = inst.classValue() - yBAcu;
            Instance instanceWithResidual = getSubInstance(inst, 1, subSpacesForEnsemble.get(m));
            instanceWithResidual.setClassValue(r);
            ensemble.get(m).trainOnInstance(instanceWithResidual);
            m_1_Instance = instanceWithResidual;
        }
    }
    public void trainBoosterUsingOtherLoss(Instance inst){
//        double[] groundTruth = new double[] {inst.classValue()};
        double[] groundTruth = new double[inst.numClasses()];
        groundTruth[(int) inst.classValue()] = 1.0;

        Instance subInstance;
        DoubleVector rawScore = new DoubleVector(StreamingGradientTreePredictor.getScoresWhenNullTree(inst, true));
        for (int m = 0; m < ensemble.size(); m++) {
            GradHess[] gradHess = mObjective.computeDerivatives(groundTruth, rawScore.getArrayRef());

//            subInstance = inst;
            subInstance = getSubInstance(inst, 1, subSpacesForEnsemble.get(m));
            ensemble.get(m).trainOnInstanceImpl(subInstance, gradHess, rawScore.getArrayRef());
            DoubleVector currentScore = new DoubleVector(ensemble.get(m).getScoresForInstance(subInstance));
            currentScore.scaleValues(learningRateOption.getValue());
            rawScore.addValues(currentScore);
        }
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
        trainBoosterUsingOtherLoss(inst);
    }

    public double[] getVotesForInstanceSquaredLoss(Instance inst) {
        double proba = 0.0;
        double[] votes;

        for (int i = 0; i < ensemble.size(); i++) {
            Instance subInstance = getSubInstance(inst, 1, subSpacesForEnsemble.get(i));
            proba += ensemble.get(i).getVotesForInstance(subInstance)[0];
        }

        votes = new double[]{1 - proba, proba};
        return votes;
    }

    public DoubleVector getRawScoreForInstanceOtherLoss(Instance inst) {
        Instance subInstance;
        DoubleVector rawScore = new DoubleVector(StreamingGradientTreePredictor.getScoresWhenNullTree(inst, true));
        for (int m = 0; m < ensemble.size(); m++) {
            subInstance = getSubInstance(inst, 1, subSpacesForEnsemble.get(m));
//            subInstance = inst;
            rawScore.addValues(ensemble.get(m).getScoresForInstance(subInstance));
        }
        return rawScore;
    }
    public double[] getVotesForInstanceOtherLoss(Instance inst) {
        return mObjective.transfer(getRawScoreForInstanceOtherLoss(inst).getArrayCopy());
    }

    public double[] getVotesForInstance(Instance inst) {
        if (ensemble == null) {
            initEnsemble(inst);
        }
        return getVotesForInstanceOtherLoss(inst);
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        // TODO Auto-generated method stub
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
//        Measurement[] measurements = null;
        return null;
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
