package com.henrygouk.sgt;

import java.io.Serializable;
import java.util.stream.IntStream;

public class SquaredError extends Objective implements Serializable {

	private static final long serialVersionUID = 1L;

    @Override
    public double[] transfer(double[] raw) {
        double[] result = new double[raw.length + 1];
//        double[] result = new double[raw.length];

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

    @Override
	public GradHess[] computeDerivatives(double[] groundTruth, double[] raw, boolean computeNegativeResidual, boolean clipPredictions) {
        GradHess[] result = new GradHess[raw.length];
        double[] transfered = null;
        if (computeNegativeResidual) {
        }else{
            transfered = transfer(raw);
        }

        for(int i = 0; i < result.length; i++) {
            if (computeNegativeResidual) {
                result[i] = new GradHess(raw[i] - groundTruth[i], 1.0);
            }else {
                result[i] = new GradHess(groundTruth[i] - transfered[i], 1.0);
            }
        }

        return result;
	}

//    @Override
//    public double loss (double[] groundTruth, double[] preds){
//        double [] ll = new double[preds.length];
//        IntStream.range(0, preds.length).parallel().forEach(i -> ll[i]=0.5*(groundTruth[i]-preds[i])*(groundTruth[i]-preds[i]));
//        double l = 0.0;
//        for (int i=0; i < ll.length; i++){
//            l += ll[i];
//        }
//        return l;
//    }

}