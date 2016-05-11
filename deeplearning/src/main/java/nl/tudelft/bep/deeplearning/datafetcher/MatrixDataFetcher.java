package nl.tudelft.bep.deeplearning.datafetcher;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.util.MathUtils;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher;

/**
 * A DataFetcher for Matrixes that are loaded from three files: .meta, .dat and
 * .lab.
 */
public class MatrixDataFetcher extends BaseDataFetcher {
	protected static final long TRAIN_SEED = 63775512; // Should never change
	protected int[] order;
	protected Random random;

	protected int[] label;
	protected DataSet[] data;
	protected int width;
	protected int height;
	protected String seperator = ",";
	protected int startIndex;

	/**
	 * Initialize a MatrixDataFetcher
	 * 
	 * @param fileName
	 *            The shared file name for the meta, lab and dat file to load,
	 *            located in the resources folder
	 * @param start
	 *            The percentage of examples to skip from the left
	 * @param end
	 *            The percentage of examples to skip from the right
	 */
	public MatrixDataFetcher(DataPath data, double start, double end) {
		List<DataSet> dataList = data.getSubset(start, end);

		this.data = dataList.toArray(new DataSet[dataList.size()]);

		this.totalExamples = dataList.size();
		this.inputColumns = data.getWidth() * data.getHeight();
		this.numOutcomes = data.getNumOutcomes();

		order = new int[totalExamples];
		for (int i = 0; i < order.length; i++) {
			order[i] = i;
		}
		random = new Random(TRAIN_SEED);
		reset();
	}

	@Override
	public void fetch(int batch) {
		if (!hasMore()) {
			throw new IllegalStateException("Unable to getFromOrigin more; there are no more images");
		}
		List<DataSet> result = new ArrayList<>();
		for(int i = 0; i<batch && cursor < this.totalExamples; i++){
			result.add(data[order[cursor++]]);
		}
		initializeCurrFromList(result);
	}

	@Override
	public void reset() {
		this.cursor = 0;
		this.curr = null;
		MathUtils.shuffleArray(this.order, this.random);
	}
}
