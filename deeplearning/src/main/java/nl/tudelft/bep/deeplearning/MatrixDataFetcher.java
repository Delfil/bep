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

	private int exampleNumber;
	private int[] label;
	private double[][] data;

	public MatrixDataFetcher(String filename, boolean shuffle, long rngSeed) throws IOException {
		String images = filename + ".dat";
		String labels = filename + ".lab";
		String meta = filename + ".meta";

		numOutcomes = 5;
		cursor = 0;
		
		readData(images, labels, meta);
		
		this.shuffle = shuffle;

		order = new int[exampleNumber];

		for (int i = 0; i < order.length; i++)
			order[i] = i;
		rng = new Random(rngSeed);
		reset(); // Shuffle order
	}

	private void readData(String images, String labels, String meta) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(meta));
		//TODO: meta file load
		reader.close();
		reader = new BufferedReader(new FileReader(images));
		for(int i = 0; i < exampleNumber; i++) {
			data[i] = Arrays.stream(reader.readLine().split(",")).mapToDouble(val -> Double.parseDouble(val)).toArray();
		}
		reader.close();
		reader = new BufferedReader(new FileReader(images));
		for(int i = 0; i < exampleNumber; i++) {
			label[i] = Integer.parseInt(reader.readLine());
		}
	}

	@Override
	public void fetch(int numExamples) {
		if (!hasMore()) {
			throw new IllegalStateException("Unable to getFromOrigin more; there are no more images");
		}

		// we need to ensure that we don't overshoot the number of examples
		// total
		List<DataSet> toConvert = new ArrayList<>(numExamples);
		for (int i = 0; i < numExamples; i++, cursor++) {
			if (!hasMore()) {
				break;
			}

			double[] img = data[order[cursor]];
			INDArray in = Nd4j.create(1, img.length);
			for (int j = 0; j < img.length; j++) {
				in.putScalar(j, img[order[cursor]]);
				// byte is loaded as signed -> convert to unsigned
			}

			in.divi(255);

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
		return this.exampleNumber;
	}
}
