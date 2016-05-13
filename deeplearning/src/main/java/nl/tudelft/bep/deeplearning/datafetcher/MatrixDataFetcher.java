package nl.tudelft.bep.deeplearning.datafetcher;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
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

/**
 * A DataFetcher for Matrixes that are loaded from three files: .meta, .dat and
 * .lab.
 */
public class MatrixDataFetcher extends BaseDataFetcher {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	protected int[] order;
	protected Random random;

	protected int[] label;
	protected double[][] data;
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
	 * @param seed
	 *            The random seed
	 * @param start
	 *            The percentage of examples to skip from the left
	 * @param end
	 *            The percentage of examples to skip from the right
	 */
	public MatrixDataFetcher(String fileName, long seed, double start, double end) {
		readData(fileName, start, end);

		order = new int[totalExamples];
		for (int i = 0; i < order.length; i++)
			order[i] = i;
		random = new Random(seed);
		reset();
	}

	/**
	 * Import all data files
	 * 
	 * @param fileName
	 *            The shared file name for the meta, lab and dat file to load,
	 *            located in the resources folder
	 * @param start
	 *            The percentage of examples to skip from the left
	 * @param end
	 *            The percentage of examples to skip from the right
	 */
	protected void readData(String fileName, double start, double end) {
		try {
			readMetaFile(fileName + ".meta", start, end);
			readMatrix(fileName + ".dat");
			readLabels(fileName + ".lab");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Import the meta data file.<p>
	 * 
	 * The meta file should contain four lines with the following data:
	 * <ul>
	 * <lu>The number of examples</li>
	 * <lu>The width of the matrix</li>
	 * <lu>The height of the matrix</li>
	 * <lu>The number of different classes</li>
	 * </ul>
	 * 
	 * @param fileName
	 *            The file name for the meta file located in the resources
	 *            folder
	 * @param start
	 *            The percentage of examples to skip from the left
	 * @param end
	 *            The percentage of examples to skip from the right
	 * @throws IOException
	 *             If an I/O error occurs
	 */
	protected void readMetaFile(String fileName, double start, double end) throws IOException {
		BufferedReader reader = readFile(fileName);

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

	/**
	 * Import the data file.
	 * 
	 * @param fileName
	 *            The file name for the dat file located in the resources folder
	 * @throws IOException
	 *             If an I/O error occurs
	 */
	protected void readMatrix(String fileName) throws IOException {
		BufferedReader reader = readFile(fileName);
		for (int i = 0; i < startIndex; i++) {
			reader.readLine();
		}
		for (int i = 0; i < totalExamples; i++) {
			data[i] = (Arrays.stream(reader.readLine().split(seperator))
					.mapToDouble(val -> Math.min(1, Math.max(0, (Double.parseDouble(val) + 1) / 2))).toArray());
			if (data[i].length != width * height) {
				try {
					throw new MetaDataMatchException();
				} catch (MetaDataMatchException e) {
					e.printStackTrace();
				}
			}
		}
		reader.close();
	}

	/**
	 * Import the label file.
	 * 
	 * @param fileName
	 *            The file name for the lab file located in the resources folder
	 * @throws IOException
	 *             If an I/O error occurs
	 */
	protected void readLabels(String fileName) throws IOException {
		BufferedReader reader = readFile(fileName);
		for (int i = 0; i < startIndex; i++) {
			reader.readLine();
		}
		for (int i = 0; i < totalExamples; i++) {
			label[i] = (int) Double.parseDouble(reader.readLine());
		}
		reader.close();
	}

	/**
	 * Gives a BufferedReader, reading the desired file.
	 * 
	 * @param fileName
	 *            Name of the desired file that can be found in the resources
	 *            folder.
	 * @return A BufferedReader reading the file that belongs to the given file
	 *         name.
	 * @throws FileNotFoundException
	 *             if the named file does not exist, is a directory rather than
	 *             a regular file, or for some other reason cannot be opened for
	 *             reading.
	 */
	private BufferedReader readFile(String fileName) throws FileNotFoundException {
		return new BufferedReader(new FileReader(MatrixDataFetcher.class.getResource(fileName).getFile()));
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
