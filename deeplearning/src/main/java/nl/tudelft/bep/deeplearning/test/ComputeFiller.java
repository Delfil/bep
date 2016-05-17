package nl.tudelft.bep.deeplearning.test;

public class ComputeFiller implements CSVFiller {

	protected final int iterations;

	public ComputeFiller(int iterations) {
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
