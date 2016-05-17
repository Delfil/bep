package nl.tudelft.bep.deeplearning.datafetcher;

import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;

import nl.tudelft.bep.deeplearning.datafetcher.MatrixDataFetcher;

public class MatrixDatasetIterator extends BaseDatasetIterator {

	public MatrixDatasetIterator(DataPath data, double start, double end) {
		super(data.getBatchSize(), 0, new MatrixDataFetcher(data, start, end));
		this.numExamples = this.fetcher.totalExamples();
	}

}
