package nl.tudelft.bep.deeplearning.test;

/**
 * A CSV table filler that fills the cells with the average of the accuracy, and
 * computes this value if it isn't computed yet.
 */
public class ComputeAverageAccuracyFiller implements CSVFiller {

	protected final int iterations;

	public ComputeAverageAccuracyFiller(int iterations) {
		this.iterations = iterations;
	}

	@Override
	public String fill(String network, String data, int epochs) {
		double result = ResultUtil.getAverageAccuracy(network, data, epochs);
		System.out.println(result);
		if (new Double(result).equals(Double.NaN)) {
			new Tester(network, data).start(this.iterations, epochs);
			return Double.toString(ResultUtil.getAverageAccuracy(network, data, epochs));
		}
		return Double.toString(result);
	}

}
