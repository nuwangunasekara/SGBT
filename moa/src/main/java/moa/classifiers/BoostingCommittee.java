/*
 *    Regressor.java
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
package moa.classifiers;

import com.henrygouk.sgt.GradHess;
import com.yahoo.labs.samoa.instances.Instance;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Regressor interface for incremental regression models. It is used only in the GUI Regression Tab. 
 *
 * @author Richard Kirkby (rkirkby@cs.waikato.ac.nz)
 * @version $Revision: 7 $
 */
public abstract class BoostingCommittee extends AbstractClassifier{
    public int committeeSize = 0;
    public static double[] getScoresWhenNullTree(int outputSize){
        return new double[outputSize];
    }
    public abstract double[] getScoresForInstance(Instance inst);

    public abstract void trainOnInstanceImpl(Instance[] instances, int multipleIterationByHessianCeiling, GradHess[] gradHess);

//    public abstract void trainOnInstanceImpl(Instance[] instances);
    public abstract void trainOnInstanceImpl(Instance inst, GradHess[] gradHess, double[] raw /* only for semiSupervisedOption */);

    public abstract ArrayList<ArrayList<HashMap<String,String>>> getCommitteeInfo();
}
