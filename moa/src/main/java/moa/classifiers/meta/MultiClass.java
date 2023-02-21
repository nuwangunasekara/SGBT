/*
 *    Perceptron.java
 *    Copyright (C) 2009 University of Waikato, Hamilton, New Zealand
 *    @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
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

import com.github.javacliparser.FlagOption;
import com.yahoo.labs.samoa.instances.Attribute;
import com.yahoo.labs.samoa.instances.DenseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.ObjectRepository;
import moa.core.Utils;
import moa.options.ClassOption;
import moa.tasks.TaskMonitor;
import org.openjdk.jol.info.GraphLayout;
import org.openjdk.jol.vm.VM;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.stream.IntStream;

import static moa.core.Measurement.getMeasurementNamed;

/**
 * Generic Multi class classifier.
 *
 * <p>Performs classic multiclass learning via committee of binary classifiers.</p>
 *
 * <p>Parameters:</p> <ul> <li>-l : base learner </li> </ul>
 *
 * @author Albert Bifet (abifet at cs dot waikato dot ac dot nz)
 * @version $Revision: 7 $
 */

public class MultiClass extends AbstractClassifier  implements MultiClassClassifier {

    private static final long serialVersionUID = 221L;

    @Override
    public String getPurposeString() {
        return "Perceptron classifier: Single perceptron classifier.";
    }

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train on instances.", Classifier.class, "meta.StreamingRandomPatches");


    Classifier baseLearner = null;
    protected Classifier[] treesCommittee;

    protected boolean reset;

    protected int numberClasses;


    @Override
    public void resetLearningImpl() {
        this.reset = true;
    }

    static public Instance newBinaryClassInstance(Instance instance){
        int classIndex = instance.classIndex();
        ArrayList<Attribute> attributes = new ArrayList<>();
        ArrayList<Double> v = new ArrayList<>();
        List<String> classAttributeValues = new ArrayList<String>();

        // set attributes and values for all except class attribute
        for (int i = 0; i < instance.numAttributes(); i++){
            if (i != classIndex){
                attributes.add(instance.attribute(i));
                v.add(Double.valueOf(instance.value(i)));
            }
        }

        // set class information
        classAttributeValues.add(0,"0");
        classAttributeValues.add(1,"1");
        Attribute classAttribute = new Attribute("classAttribute", classAttributeValues);

        attributes.add(classAttribute);
        v.add(Double.valueOf(0.0));

        Instances newInstances = new Instances("Copy", attributes, 100);
        newInstances.setClassIndex(newInstances.numAttributes()-1);

        double[] values = v.stream()
//                .parallel()
                .mapToDouble(Double::doubleValue).toArray();
        double weight = instance.weight();

        DenseInstance newInstance = new DenseInstance(weight, values);
        newInstance.setWeight(weight);
        newInstance.setDataset(newInstances);

        newInstances.add(newInstance);
        return newInstance;
    }

    @Override
    public void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {

        monitor.setCurrentActivity("Materializing learner " + (0) + "...", -1.0);

        super.randomSeedOption.setValue(this.randomSeedOption.getValue());
        super.prepareForUseImpl(monitor, repository);

        baseLearner = (Classifier) baseLearnerOption.materializeObject(monitor, repository);
        baseLearner.setRandomSeed(this.randomSeedOption.getValue());

        if (monitor.taskShouldAbort()) {
            return;
        }

        monitor.setCurrentActivity("Preparing learner " + (0) + "...", -1.0);

        baseLearner.prepareForUse(monitor, repository);
    }
    protected  void createBaseLearners(int numTrees)
    {
        treesCommittee = new Classifier[numTrees];
        for (int i = 0; i < treesCommittee.length; i++) {
            treesCommittee[i] = baseLearner.copy();
        }
    }

    Instance[] getBinaryClassInstanceArray(Instance inst){
        int actualClass = (int) inst.classValue();

        // create binaryClassInstanceArray
        Instance binaryInstance = newBinaryClassInstance(inst);

        // generate multiple instances
        Instance[] binaryClassInstanceArray = new Instance[treesCommittee.length];
        IntStream.range(0, treesCommittee.length)
//                .parallel()
//                .mapToObj(i -> binaryInstance.copy()).toArray(Instance[]::new);
                .forEach(i -> binaryClassInstanceArray[i] = binaryInstance.copy());

        // set label based on actualClass binaryClassInstanceArray
        IntStream.range(0, treesCommittee.length)
//                .parallel()
                .forEach(i -> binaryClassInstanceArray[i].setClassValue((i == actualClass) ? 1.0 : 0.0));

        return binaryClassInstanceArray;
    }
    static void trainClassifierUsingBinaryClassInstance(Instance binaryInstance, int committeeIndex,  Classifier c) {
        int actualClass = (int) binaryInstance.classValue();

        // set label based on actualClass for
        Instance bInstance = binaryInstance.copy();
        bInstance.setClassValue((committeeIndex == actualClass) ? 1.0 : 0.0);

        c.trainOnInstance(bInstance);
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {

        //Init
        if (this.reset == true) {
            this.reset = false;
            this.numberClasses = inst.numClasses();
            createBaseLearners(this.numberClasses == 2 ? 1 : this.numberClasses);
        }

        if (this.numberClasses == 2){
            treesCommittee[0].trainOnInstance(inst);
        }else {
            Instance[] binaryClassInstanceArray = getBinaryClassInstanceArray(inst);
            // create binaryClassInstanceArray
//        Instance binaryInstance = newBinaryClassInstance(inst);

            // train each learner
//        IntStream.range(0, treesCommittee.length)
//                .parallel()
//                .forEach(i -> trainClassifierUsingBinaryClassInstance(binaryInstance, i, treesCommittee[i]));

            IntStream.range(0, treesCommittee.length)
                    .parallel()
                    .forEach(i -> treesCommittee[i].trainOnInstance(binaryClassInstanceArray[i]));
        }

    }


    static double getVoteForPositiveClass(Classifier c, Instance inst){
        DoubleVector vote = new DoubleVector(c.getVotesForInstance(inst));
        if (vote.sumOfValues() > 0.0) {
            vote.normalize();
        }
        int numOfClassValues = vote.numValues();
        if (numOfClassValues > 0){
            if (numOfClassValues == 1){ // sometimes you would only get one vote, if the base learner has only seen one class
                return vote.getArrayRef()[0];
            }else{
                return vote.getArrayRef()[1];
            }
        }else{
            return 0.0;
        }
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        double[] votes = new double[inst.numClasses()];
        if (this.reset == false) {
            if (this.numberClasses == 2){
                return treesCommittee[0].getVotesForInstance(inst);
            }else {
                Instance[] binaryClassInstanceArray = getBinaryClassInstanceArray(inst);
//            Instance binaryClassInstance = newBinaryClassInstance(inst);

                // get prediction from each base learner
//            IntStream.range(0, treesCommittee.length)
//                    .parallel()
//                    .forEach(i -> votes[i] = getVoteForPositiveClass(treesCommittee[i], binaryClassInstance));

                IntStream.range(0, treesCommittee.length)
                        .parallel()
                        .forEach(i -> votes[i] = getVoteForPositiveClass(treesCommittee[i], binaryClassInstanceArray[i]));

                if (Utils.sum(votes) > 0.0) {
                    try {
                        Utils.normalize(votes);
                    } catch (Exception e) {
                        System.out.println("Error");
                        // ignore all zero votes error
                    }
                }
            }
        }
        return votes;
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
        double avgSkipCount = 0.0;
        if (treesCommittee != null) {
            Measurement[][] m = new Measurement[treesCommittee.length][];
            IntStream.range(0, treesCommittee.length)
                    .parallel()
                    .forEach(i -> m[i] = getModelMeasurementsS(treesCommittee[i]));

            for (int i = 0; i < treesCommittee.length; i++) {
                avgNumNodes += getMeasurementNamed("avgNumNodes", m[i]).getValue();
                avgSplitsByConfidence += getMeasurementNamed("avgSplitsByConfidence", m[i]).getValue();
                avgSplitsByHBound += getMeasurementNamed("avgSplitsByHBound", m[i]).getValue();
                avgSplitsByHBoundSmallerThanTieThreshold += getMeasurementNamed("avgSplitsByHBoundSmallerThanTieThreshold", m[i]).getValue();
                avgTotalSplits += getMeasurementNamed("avgTotalSplits", m[i]).getValue();
                avgSkipCount += getMeasurementNamed("skipCount", m[i]).getValue();
            }

            avgNumNodes /= (1.0 * treesCommittee.length);
            avgSplitsByConfidence /= (1.0 * treesCommittee.length);
            avgSplitsByHBound /= (1.0 * treesCommittee.length);
            avgSplitsByHBoundSmallerThanTieThreshold /= (1.0 * treesCommittee.length);
            avgTotalSplits /= (1.0 * treesCommittee.length);
            avgSkipCount /= (1.0 * treesCommittee.length);
        }

        return new Measurement[]{
                new Measurement("avgNumNodes", avgNumNodes),
                new Measurement("avgSplitsByConfidence", avgSplitsByConfidence),
                new Measurement("avgSplitsByHBound", avgSplitsByHBound),
                new Measurement("avgSplitsByHBoundSmallerThanTieThreshold", avgSplitsByHBoundSmallerThanTieThreshold),
                new Measurement("avgTotalSplits", avgTotalSplits),
                new Measurement("avgSkipCount", avgSkipCount)
        };
    }

    @Override
    public long measureByteSize() {
        long b = 0;
        // get shallow size of this
        b = VM.current().sizeOf(this);

        if (treesCommittee != null) {
            long[] byteSize = new long[treesCommittee.length];
            // get deep size of each item
            if (treesCommittee.length == 1) {
                IntStream.range(0, treesCommittee.length)
                        .forEach(i -> byteSize[i] = treesCommittee[i].measureByteSize());
            }else{
                IntStream.range(0, treesCommittee.length)
                        .parallel()
                        .forEach(i -> byteSize[i] = treesCommittee[i].measureByteSize());
            }
            for (int i = 0; i < treesCommittee.length; i++) {
                b += byteSize[i];
            }
        }
        return b;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }
}
