package nl.tudelft.bep.deeplearning;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.util.MathUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.fetcher.BaseDataFetcher;
import org.nd4j.linalg.factory.Nd4j;

public class MatrixDataFetcher extends BaseDataFetcher {
	private int[] order;
	private Random rng;
	private boolean shuffle;

	private int[] label;
	private double[][] data;

	public MatrixDataFetcher(String filename, boolean shuffle, long rngSeed, int width, int height, boolean train,
			double trainSize) throws IOException {
		String images = filename + ".dat";
		String labels = filename + ".lab";
		String meta = filename + ".meta";

		cursor = 0;
		inputColumns = width * height;

		readData(images, labels, meta, train, trainSize);

		this.shuffle = shuffle;

		order = new int[totalExamples];

		for (int i = 0; i < order.length; i++)
			order[i] = i;
		rng = new Random(rngSeed);
		reset(); // Shuffle order
	}

	private void readData(String images, String labels, String meta, boolean train, double trainSize)
			throws IOException {
		final String splitter = ","; // " "
		BufferedReader reader = new BufferedReader(new FileReader(MatrixDataFetcher.class.getResource(meta).getFile()));
		reader.readLine(); // Random text
		totalExamples = Integer.parseInt(reader.readLine());
		int skip;
		if (train) {
			skip = 0;
			totalExamples *= trainSize;
		} else {
			skip = (int) (trainSize * totalExamples);
			totalExamples *= (1 - trainSize);
		}

		int dataSize = Integer.parseInt(reader.readLine());
		numOutcomes = Integer.parseInt(reader.readLine());

		int imageSize = (int) Math.ceil(Math.sqrt(dataSize));
		imageSize *= imageSize;
		data = new double[totalExamples][imageSize];
		label = new int[totalExamples];

		reader.close();
		reader = new BufferedReader(new FileReader(MatrixDataFetcher.class.getResource(images).getFile()));
		for (int i = 0; i < skip; i++) {
			reader.readLine();
		}
		for (int i = 0; i < totalExamples; i++) {
			data[i] = (Arrays.stream(reader.readLine().split(splitter))
					.mapToDouble(val -> Math.min(1, Math.max(0, (Double.parseDouble(val) + 1) / 2))).toArray());
		}
		reader.close();
		reader = new BufferedReader(new FileReader(MatrixDataFetcher.class.getResource(labels).getFile()));
		for (int i = 0; i < skip; i++) {
			reader.readLine();
		}
		for (int i = 0; i < totalExamples; i++) {
			label[i] = (int) Double.parseDouble(reader.readLine()); // -1
		}
	}

	@Override
	public void fetch(int numExamples) {
		if (!hasMore()) {
			throw new IllegalStateException("Unable to getFromOrigin more; there are no more images");
		}

		List<DataSet> toConvert = new ArrayList<>(numExamples);

		for (int i = 0; i < numExamples; i++, cursor++) {
			if (!hasMore()) {
				break;
			}

			INDArray in = Nd4j.create(1, data[order[cursor]].length);
			for (int j = 0; j < data[order[cursor]].length; j++) {
				in.putScalar(j, data[order[cursor]][j]);
			}

			INDArray out = createOutputVector(label[order[cursor]]);
			toConvert.add(new DataSet(in, out));
		}
		initializeCurrFromList(toConvert);
	}

	@Override
	public void reset() {
		cursor = 0;
		curr = null;
		if (shuffle)
			MathUtils.shuffleArray(order, rng);
	}

	public int getNumberExamples() {
		return this.totalExamples;
	}

	public int getOutputNum() {
		return numOutcomes;
	}
}
