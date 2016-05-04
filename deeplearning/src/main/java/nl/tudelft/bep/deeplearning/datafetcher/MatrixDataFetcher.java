package nl.tudelft.bep.deeplearning.datafetcher;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.lang.reflect.Array;
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
	protected int[] order;
	protected Random random;

	protected int[] label;
	protected double[][] data;
	protected int width;
	protected int height;
	protected String splitter = ",";
	protected int startIndex;

	public MatrixDataFetcher(String fileName, long seed, double start, double end)
			throws IOException {
		readData(fileName, start, end);
		
		order = new int[totalExamples];
		for (int i = 0; i < order.length; i++)
			order[i] = i;
		random = new Random(seed);
		reset();
	}

	protected void readData(String fileName, double start, double end) throws IOException {
		readMetaFile(fileName + ".meta", start, end);
		readMatrix(fileName + ".dat");
		readLabels(fileName + ".lab");
	}

	protected void readMetaFile(String fileName, double start, double end) throws IOException {
		BufferedReader reader = new BufferedReader(
				new FileReader(MatrixDataFetcher.class.getResource(fileName).getFile()));

		int examples = Integer.parseInt(reader.readLine());

		startIndex = (int) (examples * start);
		int endIndex = (int) (examples * end);
		totalExamples = endIndex - startIndex;

		width = Integer.parseInt(reader.readLine());
		height = Integer.parseInt(reader.readLine());
		numOutcomes = Integer.parseInt(reader.readLine());

		inputColumns = width * height;

		data = new double[totalExamples][width * height];
		label = new int[totalExamples];

		reader.close();
	}

	protected void readMatrix(String fileName) throws IOException {
		BufferedReader reader = new BufferedReader(
				new FileReader(MatrixDataFetcher.class.getResource(fileName).getFile()));
		for (int i = 0; i < startIndex; i++) {
			reader.readLine();
		}
		for (int i = 0; i < totalExamples; i++) {
			data[i] = (Arrays.stream(reader.readLine().split(splitter))
					.mapToDouble(val -> Math.min(1, Math.max(0, (Double.parseDouble(val) + 1) / 2))).toArray());
			data[i] = Arrays.copyOf(data[i], width*height);
		}
		reader.close();
	}

	protected void readLabels(String fileName) throws IOException {
		BufferedReader reader = new BufferedReader(
				new FileReader(MatrixDataFetcher.class.getResource(fileName).getFile()));
		for (int i = 0; i < startIndex; i++) {
			reader.readLine();
		}
		for (int i = 0; i < totalExamples; i++) {
			label[i] = (int) Double.parseDouble(reader.readLine()) - 1;
		}
		reader.close();
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
		MathUtils.shuffleArray(order, random);
	}

	public int getNumberExamples() {
		return this.totalExamples;
	}

	public int getOutputNum() {
		return numOutcomes;
	}

	public int getWidth() {
		return width;
	}
	
	public int getHeight() {
		return height;
	}
}
