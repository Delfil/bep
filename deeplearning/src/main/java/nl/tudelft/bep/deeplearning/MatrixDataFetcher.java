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

	public MatrixDataFetcher(String filename, boolean shuffle, long rngSeed) throws IOException {
		String images = filename + ".dat";
		String labels = filename + ".lab";
		String meta = filename + ".meta";

		numOutcomes = 5;
		cursor = 0;
		inputColumns = 113 * 113;

		readData(images, labels, meta);

		this.shuffle = shuffle;

		order = new int[totalExamples];

		for (int i = 0; i < order.length; i++)
			order[i] = i;
		rng = new Random(rngSeed);
		reset(); // Shuffle order
	}

	private void readData(String images, String labels, String meta) throws IOException {
		final String splitter = ","; //"   "
		BufferedReader reader = new BufferedReader(new FileReader(meta));
		reader.readLine(); // Random text
		totalExamples = Integer.parseInt(reader.readLine());
		totalExamples -= totalExamples % 64;
		int dataSize = Integer.parseInt(reader.readLine()); // Number of
															// elements
		this.numOutcomes = Integer.parseInt(reader.readLine());

		int imageSize = (int) Math.ceil(Math.sqrt(dataSize));
		imageSize *= imageSize;
		data = new double[totalExamples][imageSize];
		label = new int[totalExamples];

		reader.close();
		reader = new BufferedReader(new FileReader(images));
		for (int i = 0; i < totalExamples; i++) {
			data[i] = (Arrays.stream(reader.readLine().substring(3).split(splitter)).mapToDouble(val -> (Double.parseDouble(val.split("e")[0])+1)/2).toArray());
		}
		reader.close();
		reader = new BufferedReader(new FileReader(labels));
		for (int i = 0; i < totalExamples; i++) {
			label[i] = (int)Double.parseDouble(reader.readLine().split("e")[0]) - 1;
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

			INDArray in = Nd4j.create(1, data[order[cursor]].length);
			for (int j = 0; j < data[order[cursor]].length; j++) {
				in.putScalar(j, data[order[cursor]][j]);
				// byte is loaded as signed -> convert to unsigned
//				if(j%28==0) System.out.println();
//				if(data[order[cursor]][j] == 1.0)
//					System.out.print("O");
//				else
//					System.out.print(" ");
			}
//			System.out.println();

//			in.divi(255.0);
			INDArray out = createOutputVector(label[order[cursor]]);
			toConvert.add(new DataSet(in, out));
			System.out.println(in);
			System.out.println(out);
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
}
