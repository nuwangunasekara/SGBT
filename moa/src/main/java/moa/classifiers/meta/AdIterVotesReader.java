/*
 *    kNN.java
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


import com.github.javacliparser.FileOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.capabilities.Capability;
import moa.capabilities.ImmutableCapabilities;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.MultiClassClassifier;
import moa.core.Measurement;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;


public class AdIterVotesReader extends AbstractClassifier implements MultiClassClassifier {

    private static final long serialVersionUID = 1L;
	FileInputStream fileInputStream = null;
	private BufferedReader inputReader = null;

	private long instancesSeen = 0;

	public FileOption votesFileOption = new FileOption("votesFile", 'f',
			"Votes File.", null, "csv", true);

	public IntOption skipFirstNInstances = new IntOption(
			"skipFirstNInstances",
			'S',
			"skipFirstNInstances",
			0, 0, Integer.MAX_VALUE);

	public IntOption ensembleSize = new IntOption(
			"ensembleSize",
			's',
			"ensembleSize",
			1, 0, Integer.MAX_VALUE);


	private void initReader(){
		try {
			fileInputStream = new FileInputStream(votesFileOption.getFile());
			inputReader = new BufferedReader(new InputStreamReader(fileInputStream));
		} catch (Exception e) {
			System.out.println(e.getMessage());
			System.exit(1);
		}
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		double [] v;
		if (fileInputStream == null){
			initReader();
		}
		instancesSeen++;
		if (instancesSeen <= skipFirstNInstances.getValue()){
			double vv = this.classifierRandom.nextDouble();
			return new double[]{1.0 - vv, vv};
		}
		// read out a line
//		int labelFromArff = inst.attribute(inst.classIndex()).indexOfValue("DOWN");
		List<Double> votes = new ArrayList<>();
		try {
			// read the data line
			String inputFileLine = inputReader.readLine();
			String[] labelAndVotes = inputFileLine.split(",");
			for (int i=0; i< labelAndVotes.length-1; i++){
				votes.add(Double.valueOf(labelAndVotes[i]));
			}
			// read out the blank line
//			inputFileLine = inputReader.readLine();
		} catch (Exception e) {
			System.out.println(e.getMessage() + " " +instancesSeen);
			System.exit(1);
		}
		v = votes.stream().mapToDouble(Double::doubleValue).toArray();
		return v;
	}

	@Override
	public String getPurposeString() {
		return "Read votes from a file";
	}

	@Override
	public void setModelContext(InstancesHeader context) { }

	@Override
	public void trainOnInstanceImpl(Instance inst) { }

    @Override
    public void resetLearningImpl() { }

	@Override
	public ImmutableCapabilities defineImmutableCapabilities() {
		return new ImmutableCapabilities(Capability.VIEW_STANDARD, Capability.VIEW_LITE);
	}

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) { }

    public boolean isRandomizable() {
        return true;
    }
}