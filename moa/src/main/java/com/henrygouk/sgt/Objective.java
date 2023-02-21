package com.henrygouk.sgt;

public abstract class Objective {
//    double[] predictions;
    public double lossForAllClasses = 0.0;
    public abstract GradHess[] computeDerivatives(double[] groundTruth, double[] raw, boolean computeNegativeResidual, boolean clipPredictions);

    public double[] transfer(double[] raw) {
        return raw;
    }

//    public double loss (double[] groundTruth) {return 0.0;}
//    public double loss (double[] groundTruth, double[] preds) {return 0.0;}
}