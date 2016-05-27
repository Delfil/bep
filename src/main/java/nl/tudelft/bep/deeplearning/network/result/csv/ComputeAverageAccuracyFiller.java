package nl.tudelft.bep.deeplearning.network.result.csv;

import nl.tudelft.bep.deeplearning.network.result.EvaluationFileUtil;
import nl.tudelft.bep.deeplearning.network.result.ResultUtil;
import nl.tudelft.bep.deeplearning.network.result.Tester;

/**
 * A CSV table filler that fills the cells with the average of the accuracy, and
 * computes this value if it isn't computed yet.
 */
public class ComputeAverageAccuracyFiller implements CSVFiller {

	private final int iterations;

	/**
	 * Initialize a {@link ComputeAverageAccuracyFiller}.
	 * 
	 * @param iterations
	 *            the number of iterations to make if a value isn't computed yet
	 */
	public ComputeAverageAccuracyFiller(final int iterations) {
		this.iterations = iterations;
	}

	@Override
	public String fill(final String network, final String data, final int epochs) {
		double result = ResultUtil.getAverageAccuracy(network, data, epochs);
		if (new Double(result).equals(Double.NaN)
				|| EvaluationFileUtil.load(epochs, data, network).size() < this.iterations) {
			new Tester(network, data).start(this.iterations, epochs);
			return Double.toString(ResultUtil.getAverageAccuracy(network, data, epochs));
		}
		return Double.toString(result);
	}

	/**
	 * @return the iterations
	 */
	public int getIterations() {
		return this.iterations;
	}

}
