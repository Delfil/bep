package nl.tudelft.bep.deeplearning.test;

/**
 * A CSV table filler that fills the cells with the average of the accuracy, and
 * ignores this value if it isn't computed yet.
 */
public class IgnoreAverageAccuracyFiller implements CSVFiller {

	@Override
	public String fill(String network, String data, int epochs) {
		return Double.toString(ResultUtil.getAverageAccuracy(network, data, epochs));
	}

}
