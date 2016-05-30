package nl.tudelft.bep.deeplearning.network.result.csv;

import nl.tudelft.bep.deeplearning.network.result.ResultUtil;

/**
 * A CSV table filler that fills the cells with the average of the accuracy, and
 * ignores this value if it isn't computed yet.
 */
public class IgnoreAverageAccuracyFiller implements CSVFiller {

	@Override
	public String fill(final String network, final String data, final int epochs) {
		return Double.toString(ResultUtil.getAverageAccuracy(network, data, epochs));
	}

}
