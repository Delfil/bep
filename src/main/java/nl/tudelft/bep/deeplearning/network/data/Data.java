package nl.tudelft.bep.deeplearning.network.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.apache.commons.lang.ArrayUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

import nl.tudelft.bep.deeplearning.network.data.exception.MetaDataMatchException;
import nl.tudelft.bep.deeplearning.network.data.exception.UnknownMetaDataFileVersion;

public class Data {
	protected static final String META_SUFFIX = ".meta";
	protected static final String DATA_SUFFIX = ".dat";
	protected static final String LABEL_SUFFIX = ".lab";
	protected static final String SEPERATOR = ",";
	public static final String DATA_FOLDER = "datasets";
	protected static final String F = "/";

	private final String path;
	private final String name;
	private final int version;
	private final long timeStamp;
	private final int examples;
	private final int width;
	private final int height;
	private final int numOutcomes;
	private final int batchSize;
	private final double trainPercentage;
	private DataSet[][] data;

	protected Data(final String path, final String name, final int version, final long timeStamp, final int examples,
			final int width, final int height, final int numOutcomes, final int batchSize,
			final double trainPercentage) {
		this.path = path;
		this.name = name;
		this.version = version;
		this.timeStamp = timeStamp;
		this.examples = examples;
		this.width = width;
		this.height = height;
		this.numOutcomes = numOutcomes;
		this.batchSize = batchSize;
		this.trainPercentage = trainPercentage;
	}

	/**
	 * Construct a {@link Data} from the files in the given path.
	 * 
	 * @param folderName
	 *            the name of the folder to read
	 * 
	 * @return a {@link Data} constructed from the files in the given path
	 */
	public static Data readDataSet(final String folderName) {
		Data data = null;
		try {
			data = readMetaFile(folderName);
			data.importData();
		} catch (IOException | NumberFormatException | UnknownMetaDataFileVersion | MetaDataMatchException e) {
			e.printStackTrace();
		}
		return data;
	}

	/**
	 * Imports the data from the instance its files to its memory.
	 * 
	 * @throws IOException
	 *             If an I/O error occurs
	 * @throws MetaDataMatchException
	 *             if the meta data and dat data doesn't match
	 */
	protected void importData() throws IOException, MetaDataMatchException {
		double[][] matrix = this.readMatrices();
		int[] labels = this.readLabels();

		@SuppressWarnings("unchecked")
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
	protected static Data readMetaFile(final String name)
			throws UnknownMetaDataFileVersion, NumberFormatException, IOException {
		String pathName = DATA_FOLDER + F + name;
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
			return new Data(pathName, name, version, timeStamp, examples, width, height, numOutcomes, batchSize,
					trainPercentage);
		} else {
			throw new UnknownMetaDataFileVersion();
		}
	}

	/**
	 * Finds a file with the given suffix in the given path.
	 * 
	 * @param pathName
	 *            the path to search in
	 * @param suffix
	 *            the suffix to search for
	 * @return a {@link BufferedReader}, reading the found file
	 */
	protected static BufferedReader findFile(final String pathName, final String suffix) {
		File dir = new File(pathName);
		if (!dir.exists()) {
			return null;
		}
		for (File file : dir.listFiles()) {
			if (file.getName().endsWith(suffix)) {
				try {
					return new BufferedReader(new FileReader(file));
				} catch (FileNotFoundException e) {
					e.printStackTrace();
				}
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
	 * @throws MetaDataMatchException
	 *             if the meta data and dat data doesn't match
	 */
	protected double[][] readMatrices() throws IOException, MetaDataMatchException {
		BufferedReader reader = findFile(this.path, DATA_SUFFIX);

		double[][] matrix = new double[this.examples][];

		for (int i = 0; i < this.examples; i++) {
			List<Double> al = new ArrayList<>();
			List<Double> left = new ArrayList<Double>();
			Arrays.stream(reader.readLine().split(SEPERATOR))
					.mapToDouble(val -> Math.min(1, Math.max(0, (Double.parseDouble(val) + 1) / 2)))
					.forEach(a -> left.add(a));
			al.addAll(left);
//			left.stream().map(d -> 1.0 - d).forEach(a -> al.add(a));
			matrix[i] = ArrayUtils.toPrimitive(al.toArray(new Double[al.size()]));
			if (matrix[i].length != this.width * this.height) {
				throw new MetaDataMatchException();
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
		BufferedReader reader = findFile(this.path, LABEL_SUFFIX);
		int[] label = new int[this.examples];
		for (int i = 0; i < this.examples; i++) {
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
	public List<DataSet> getSubset(final double start, final double end) {
		List<DataSet> subset = new ArrayList<>();
		for (int i = 0; i < this.numOutcomes; i++) {
			int size = this.data[i].length;
			for (int j = (int) (size * start), stop = (int) (size * end); j < stop; j++) {
				subset.add(this.data[i][j]);
			}
		}
		return subset;
	}

	public String getPath() {
		return this.path;
	}

	public int getVersion() {
		return this.version;
	}

	public long getTimeStamp() {
		return this.timeStamp;
	}

	public int getExamples() {
		return this.examples;
	}

	public int getWidth() {
		return this.width;
	}

	public int getHeight() {
		return this.height;
	}

	public int getNumOutcomes() {
		return this.numOutcomes;
	}

	public int getBatchSize() {
		return this.batchSize;
	}

	public double getTrainPercentage() {
		return this.trainPercentage;
	}

	public String getName() {
		return this.name;
	}

}
