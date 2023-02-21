package com.henrygouk.sgt;

import java.io.Serializable;

public class SoftmaxCrossEntropy extends Objective implements Serializable {

    private static final long serialVersionUID = 1L;

    @Override
    public GradHess[] computeDerivatives(double[] groundTruth, double[] raw, boolean computeNegativeResidual, boolean clipPredictions) {
        GradHess[] result = new GradHess[raw.length];
        super.lossForAllClasses = 0.0;
        double[] predictions = transfer(raw);

        for(int i = 0; i < result.length; i++) {
            if (clipPredictions){
                predictions[i] = Math.max(predictions[i] , 0.0001);
                predictions[i] = Math.min(predictions[i] , 0.9999);
            }
            if (computeNegativeResidual){
                result[i] = new GradHess(predictions[i] - groundTruth[i], predictions[i] * (1.0 - predictions[i]));
            }else{
                result[i] = new GradHess(groundTruth[i] - predictions[i], predictions[i] * (1.0 - predictions[i]));
            }
            super.lossForAllClasses +=  -groundTruth[i] * Math.log(predictions[i]);
        }

        return result;
    }
    
    @Override
    public double[] transfer(double[] raw) {
        double[] result = new double[raw.length + 1];

        for(int i = 0; i < raw.length; i++) {
            result[i] = raw[i];
        }

        double max = Double.NEGATIVE_INFINITY;
        double sum = 0.0;

        for(int i = 0; i < result.length; i++) {
            max = Math.max(max, result[i]);
        }

        for(int i = 0; i < result.length; i++) {
            result[i] = Math.exp(result[i] - max);
            sum += result[i];
        }

        for(int i = 0; i < result.length; i++) {
            result[i] /= sum;
        }

        return result;
    }

//    @Override
//    public double loss (double[] groundTruth){
//        return this.loss(groundTruth, predictions);
//    }
//
//    @Override
//    public double loss (double[] groundTruth, double[] preds){
//        double [] ll = new double[preds.length];
//        IntStream.range(0, preds.length).parallel().forEach(i -> ll[i]=groundTruth[i]*Math.log(preds[i]));
//        double l = 0.0;
//        for (int i=0; i < ll.length; i++){
//            l += ll[i];
//        }
//        return -l;
//    }
}