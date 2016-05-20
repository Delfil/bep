package nl.tudelft.bep.deeplearning.network.result.csv;

import nl.tudelft.bep.deeplearning.network.result.EvaluationFileUtil;
import nl.tudelft.bep.deeplearning.network.result.ResultUtil;
import nl.tudelft.bep.deeplearning.network.result.Tester;

public class ComputeAverageAccuracyAccurateFiller implements CSVFiller {

	final int minIterations;
	final int maxIterations;
	final double threshold;

	public ComputeAverageAccuracyAccurateFiller(int minIterations, int maxIterations, double threshold) {
		this.minIterations = minIterations;
		this.maxIterations = maxIterations;
		this.threshold = threshold;
	}

	@Override
	public String fill(String network, String data, int epochs) {
		if (EvaluationFileUtil.load(epochs, data, network).size() < this.minIterations) {
			new Tester(network, data).start(this.minIterations, epochs);
		}
		int iterations = this.minIterations;
		while (ResultUtil.getTTest(network, data, epochs) < threshold) {
			iterations += this.minIterations;
			if (iterations > this.maxIterations) {
				return Double.toString(Double.NaN);
			}
			new Tester(network, data).start(iterations, epochs);
		}
		return Double.toString(ResultUtil.getAverageAccuracy(network, data, epochs));
	}

}
