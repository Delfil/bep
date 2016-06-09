package nl.tudelft.bep.deeplearning.network.data;

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
	protected static final String SEPERATOR = ",";

	private int[] order;
	private Random random;
	private int[] label;
	private DataSet[] data;
	private int width;
	private int height;
	private int startIndex;

	/**
	 * Initialize a MatrixDataFetcher.
	 * 
	 * @param data
	 *            The {@link GeneExpressionDatabase} instance to fetch from
	 * @param start
	 *            The percentage of examples to skip from the left
	 * @param end
	 *            The percentage of examples to skip from the right
	 */
	public MatrixDataFetcher(final GeneExpressionDatabase data, final double start, final double end) {
		List<DataSet> dataList = data.getSubset(start, end);

		this.data = dataList.toArray(new DataSet[dataList.size()]);

		this.totalExamples = dataList.size();
		this.inputColumns = data.getWidth() * data.getHeight();
		this.numOutcomes = data.getNumOutcomes();

		this.order = new int[this.totalExamples];
		for (int i = 0; i < this.order.length; i++) {
			this.order[i] = i;
		}
		this.random = new Random(TRAIN_SEED);
		this.reset();
	}

	@Override
	public void fetch(final int batch) {
		if (!hasMore()) {
			throw new IllegalStateException("Unable to getFromOrigin more; there are no more images");
		}
		List<DataSet> result = new ArrayList<>();
		for (int i = 0; i < batch && cursor < this.totalExamples; i++) {
			result.add(this.data[this.order[this.cursor++]]);
		}
		initializeCurrFromList(result);
	}

	@Override
	public void reset() {
		this.cursor = 0;
		this.curr = null;
		MathUtils.shuffleArray(this.order, this.random);
	}

	protected int[] getOrder() {
		return this.order;
	}

	protected Random getRandom() {
		return this.random;
	}

	protected int[] getLabel() {
		return this.label;
	}

	protected DataSet[] getData() {
		return this.data;
	}

	protected int getWidth() {
		return this.width;
	}

	protected int getHeight() {
		return this.height;
	}

	protected int getStartIndex() {
		return this.startIndex;
	}
}
