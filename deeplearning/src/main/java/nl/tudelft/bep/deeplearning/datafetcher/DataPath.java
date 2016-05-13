package nl.tudelft.bep.deeplearning.datafetcher;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

public class DataPath {
	protected static final String META_SUFFIX = ".meta";
	protected static final String DATA_SUFFIX = ".dat";
	protected static final String LABEL_SUFFIX = ".lab";
	protected static final String SEPERATOR = ",";
	public static final String DATA_FOLDER = "datasets";
	protected static final String F = "/";

	protected final String path;
	protected final int version;
	protected final long timeStamp;
	protected final int examples;
	protected final int width;
	protected final int height;
	protected final int numOutcomes;
	protected final int batchSize;
	protected final double trainPercentage;
	protected DataSet[][] data;

	protected DataPath(String path, int version, long timeStamp, int examples, int width, int height, int numOutcomes,
			int batchSize, double trainPercentage) {
		this.path = path;
		this.version = version;
		this.timeStamp = timeStamp;
		this.examples = examples;
		this.width = width;
		this.height = height;
		this.numOutcomes = numOutcomes;
		this.batchSize = batchSize;
		this.trainPercentage = trainPercentage;
	}

	public static DataPath readDataSet(String folderName) {
		DataPath data = null;
		try {
			data = readMetaFile(DATA_FOLDER + F + folderName);
			data.importData();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return data;
	}

	protected void importData() throws IOException {
		double[][] matrix = this.readMatrices();
		int[] labels = this.readLabels();

		List<DataSet>[] toConvert = new List[this.numOutcomes];
		for (int i = 0; i < this.numOutcomes; i++) {
			toConvert[i] = new ArrayList<>(this.examples);
		}
		for (int i = 0; i < this.examples; i++) {
			INDArray in = Nd4j.create(1, this.width * this.height);
			// TODO: Test Nd4j.create(width, height)
			for (int j = 0; j < matrix[i].length; j++) {
				in.putScalar(j, matrix[i][j]);
			}

			INDArray out = FeatureUtil.toOutcomeVector(labels[i], this.numOutcomes);
			toConvert[labels[i]].add(new DataSet(in, out));
		}

		this.data = new DataSet[this.numOutcomes][];
		for (int i = 0; i < this.numOutcomes; i++) {
			this.data[i] = toConvert[i].toArray(new DataSet[toConvert[i].size()]);
		}
	}

	/**
	 * Import the meta data file.
	 * <p>
	 * 
	 * The meta file should contain four lines with the following data:
	 * <ul>
	 * <li>The data presentation version</li>
	 * <li>The time stamp of creation, used as identifier of the data, which is
	 * the difference, measured in milliseconds, between the current time and
	 * midnight, January 1, 1970 UTC.</li>
	 * <li>The number of examples</li>
	 * <li>The width of the matrix</li>
	 * <li>The height of the matrix</li>
	 * <li>The number of different classes</li>
	 * <li>Percentage of data used for training</li>
	 * <li>The batch size</li>
	 * </ul>
	 * 
	 * @param fileName
	 *            The file name for the meta file located in the resources
	 *            folder
	 * @param start
	 *            The percentage of examples to skip from the left
	 * @param end
	 *            The percentage of examples to skip from the right
	 * @throws NumberFormatException
	 * @throws IOException
	 *             If an I/O error occurs
	 */
	protected static DataPath readMetaFile(String pathName)
			throws UnknownMetaDataFileVersion, NumberFormatException, IOException {
		BufferedReader reader = new BufferedReader(findFile(pathName, META_SUFFIX));
		int version = Integer.parseInt(reader.readLine());

		if (version > 0) {
			long timeStamp = Long.parseLong(reader.readLine());
			int examples = Integer.parseInt(reader.readLine());
			int width = Integer.parseInt(reader.readLine());
			int height = Integer.parseInt(reader.readLine());
			int numOutcomes = Integer.parseInt(reader.readLine());
			double trainPercentage = Double.parseDouble(reader.readLine());
			int batchSize = Integer.parseInt(reader.readLine());
			return new DataPath(pathName, version, timeStamp, examples, width, height, numOutcomes, batchSize,
					trainPercentage);
		} else {
			throw new UnknownMetaDataFileVersion();
		}
	}

	private static BufferedReader findFile(String pathName, String suffix) {
		File dir = new File(pathName);
		for(File file : dir.listFiles()) {
			if(file.getName().endsWith(suffix))
				try {
					return new BufferedReader(new FileReader(file));
				} catch (FileNotFoundException e) {
					e.printStackTrace();
				}
		}
		return null;
	}

	/**
	 * Import the data file.
	 * 
	 * @return a matrix corresponding with the meta file
	 * @throws IOException
	 *             If an I/O error occurs
	 */
	protected double[][] readMatrices() throws IOException {
		BufferedReader reader = findFile(this.path, DATA_SUFFIX);

		double[][] matrix = new double[examples][];

		for (int i = 0; i < examples; i++) {
			matrix[i] = (Arrays.stream(reader.readLine().split(SEPERATOR))
					.mapToDouble(val -> Math.min(1, Math.max(0, (Double.parseDouble(val) + 1) / 2))).toArray());
			if (matrix[i].length != width * height) {
				try {
					throw new MetaDataMatchException();
				} catch (MetaDataMatchException e) {
					e.printStackTrace();
				}
			}
		}
		reader.close();
		return matrix;
	}

	/**
	 * Import the label file.
	 * 
	 * @return a label list corresponding with the meta file
	 * @throws IOException
	 *             If an I/O error occurs
	 */
	protected int[] readLabels() throws IOException {
		BufferedReader reader = findFile(path, LABEL_SUFFIX);
		int[] label = new int[examples];
		for (int i = 0; i < examples; i++) {
			label[i] = (int) Double.parseDouble(reader.readLine());
		}
		reader.close();
		return label;
	}

	/**
	 * Produces a subset of the data using the percentage parameters to
	 * determine what part to take from each separate label data array.
	 * 
	 * @param start
	 *            The percentage of examples to skip from the left
	 * @param end
	 *            The percentage of examples to skip from the right
	 * @return a subset of the data
	 */
	public List<DataSet> getSubset(double start, double end) {
		List<DataSet> subset = new ArrayList<>();
		for (int i = 0; i < this.numOutcomes; i++) {
			int size = data[i].length;
			for (int j = (int) (size * start), stop = (int) (size * end); j < stop; j++) {
				subset.add(data[i][j]);
			}
		}
		return subset;
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
	public static BufferedReader readFile(String fileName) throws FileNotFoundException {
		return new BufferedReader(new FileReader(new File(fileName)));
	}

	public String getPath() {
		return path;
	}

	public int getVersion() {
		return version;
	}

	public long getTimeStamp() {
		return timeStamp;
	}

	public int getExamples() {
		return examples;
	}

	public int getWidth() {
		return width;
	}

	public int getHeight() {
		return height;
	}

	public int getNumOutcomes() {
		return numOutcomes;
	}

	public int getBatchSize() {
		return batchSize;
	}

	public double getTrainPercentage() {
		return trainPercentage;
	}

}
