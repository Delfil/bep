package nl.tudelft.bep.deeplearning.datafetcher;

import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;

import nl.tudelft.bep.deeplearning.datafetcher.MatrixDataFetcher;

public class MatrixDatasetIterator extends BaseDatasetIterator {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public MatrixDatasetIterator(int batch, MatrixDataFetcher fetcher) {
		super(batch, fetcher.getNumberExamples(), fetcher);
	}


}
