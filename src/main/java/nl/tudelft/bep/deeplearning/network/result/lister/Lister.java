package nl.tudelft.bep.deeplearning.network.result.lister;

public interface Lister {
	/**
	 * Generate a list.
	 * 
	 * @param epoch
	 *            the epoch number to use
	 * @param dataSet
	 *            the data set to use
	 * @param network
	 *            the network to use
	 * @return a list in form of a string
	 */
	String list(int epoch, String dataSet, String network);
}
