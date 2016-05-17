package nl.tudelft.bep.deeplearning.test;

public interface CSVFiller {
	/**
	 * Computes a {@link String} to fill in a cell in the csv table
	 * 
	 * @param network
	 *            the row of the cell
	 * @param data
	 *            the column of the cell
	 * @param epoch
	 *            the epoch to use
	 * @return a {@link String} to fill in a cell in the csv table
	 */
	public String fill(String network, String data, int epoch);
}
