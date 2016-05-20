package nl.tudelft.bep.deeplearning.network.data;

import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;

import nl.tudelft.bep.deeplearning.network.data.MatrixDataFetcher;

public class MatrixDatasetIterator extends BaseDatasetIterator {

	/**
	 * Initialize a MatrixDatasetIterator
	 * 
	 * @param data
	 *            the {@link Data} instance to iterate over
	 * @param start
	 *            The percentage of examples to skip from the left
	 * @param end
	 *            The percentage of examples to skip from the right
	 */
	public MatrixDatasetIterator(Data data, double start, double end) {
		super(data.getBatchSize(), 0, new MatrixDataFetcher(data, start, end));
		this.numExamples = this.fetcher.totalExamples();
	}

	protected MatrixDatasetIterator(Data data, MatrixDataFetcher df) {
		super(data.getBatchSize(), 0, df);
		this.numExamples = this.fetcher.totalExamples();
	}

}
