package nl.tudelft.bep.deeplearning;

public class IgnoreFiller implements CSVFiller {

	@Override
	public String fill(String network, String data, int batchSzie) {
		return Double.toString(ResultUtil.getAverageAccuracy(network, data, batchSzie));
	}

}
