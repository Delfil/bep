package nl.tudelft.bep.deeplearning.network.data;

import java.util.List;

import org.nd4j.linalg.dataset.DataSet;

public interface GeneExpressionDatabase {

	/**
	 * Produces a subset of the data using the percentage parameters to
	 * determine what part to take from each separate label data array.
	 * 
	 * @param start
	 *            The percentage of examples to skip from the left
	 * @param end
	 *            The percentage of examples to skip from the right
	 * @return a subset of the data
	 */
	List<DataSet> getSubset(final double start, final double end);

	String getPath();

	int getVersion();

	long getTimeStamp();

	int getExamples();

	int getWidth();

	int getHeight();

	int getNumOutcomes();

	int getBatchSize();

	double getTrainPercentage();

	String getName();
}
