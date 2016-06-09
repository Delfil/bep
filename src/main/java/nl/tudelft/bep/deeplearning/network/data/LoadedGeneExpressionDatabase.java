package nl.tudelft.bep.deeplearning.network.data;

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

import nl.tudelft.bep.deeplearning.network.data.exception.MetaDataMatchException;
import nl.tudelft.bep.deeplearning.network.data.exception.UnknownMetaDataFileVersion;

public final class LoadedGeneExpressionDatabase implements GeneExpressionDatabase {
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

	/**
	 * Use the {@link LoadedGeneExpressionDatabase.Loader} to initialize a instance of
	 * this class.
	 */
	private LoadedGeneExpressionDatabase(final String path, final String name, final int version, final long timeStamp,
			final int examples, final int width, final int height, final int numOutcomes, final int batchSize,
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

	@Override
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

	@Override
	public String getPath() {
		return this.path;
	}

	@Override
	public int getVersion() {
		return this.version;
	}

	@Override
	public long getTimeStamp() {
		return this.timeStamp;
	}

	@Override
	public int getExamples() {
		return this.examples;
	}

	@Override
	public int getWidth() {
		return this.width;
	}

	@Override
	public int getHeight() {
		return this.height;
	}

	@Override
	public int getNumOutcomes() {
		return this.numOutcomes;
	}

	@Override
	public int getBatchSize() {
		return this.batchSize;
	}

	@Override
	public double getTrainPercentage() {
		return this.trainPercentage;
	}

	@Override
	public String getName() {
		return this.name;
	}

	public static final class Loader {
		private Loader() {
		}

		protected static final String META_SUFFIX = ".meta";
		protected static final String DATA_SUFFIX = ".dat";
		protected static final String LABEL_SUFFIX = ".lab";
		protected static final String SEPERATOR = ",";
		public static final String DATA_FOLDER = "datasets";
		protected static final String F = "/";

		/**
		 * Construct a {@link GeneExpressionDatabase} from the files in the
		 * given path.
		 * 
		 * @param folderName
		 *            the name of the folder to read
		 * 
		 * @return a {@link GeneExpressionDatabase} constructed from the files
		 *         in the given path
		 */
		public static GeneExpressionDatabase load(final String folderName) {
			LoadedGeneExpressionDatabase data = null;
			try {
				data = readMetaFile(folderName);
				importData(data);
			} catch (IOException | NumberFormatException | UnknownMetaDataFileVersion | MetaDataMatchException e) {
				e.printStackTrace();
			}
			return data;
		}

		/**
		 * Imports the data from the instance its files to its memory.
		 * 
		 * @param data
		 * 
		 * @throws IOException
		 *             If an I/O error occurs
		 * @throws MetaDataMatchException
		 *             if the meta data and data doesn't match
		 */
		protected static void importData(final LoadedGeneExpressionDatabase data) throws IOException, MetaDataMatchException {
			double[][] matrix = readMatrices(data);
			int[] labels = readLabels(data);

			@SuppressWarnings("unchecked")
			List<DataSet>[] toConvert = new List[data.numOutcomes];
			for (int i = 0; i < data.numOutcomes; i++) {
				toConvert[i] = new ArrayList<>(data.examples);
			}
			for (int i = 0; i < data.examples; i++) {
				INDArray in = Nd4j.create(1, data.width * data.height);
				for (int j = 0; j < matrix[i].length; j++) {
					in.putScalar(j, matrix[i][j]);
				}
				INDArray out = FeatureUtil.toOutcomeVector(labels[i], data.numOutcomes);
				toConvert[labels[i]].add(new DataSet(in, out));
			}
			data.data = new DataSet[data.numOutcomes][];
			for (int i = 0; i < data.numOutcomes; i++) {
				data.data[i] = toConvert[i].toArray(new DataSet[toConvert[i].size()]);
			}
		}

		/**
		 * Import the meta data file.
		 * <p>
		 * 
		 * The meta file should contain four lines with the following data:
		 * <ul>
		 * <li>The data presentation version</li>
		 * <li>The time stamp of creation, used as identifier of the data, which
		 * is the difference, measured in milliseconds, between the current time
		 * and midnight, January 1, 1970 UTC.</li>
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
		protected static LoadedGeneExpressionDatabase readMetaFile(final String name)
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
				return new LoadedGeneExpressionDatabase(pathName, name, version, timeStamp, examples, width, height,
						numOutcomes, batchSize, trainPercentage);
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
		 * @param data
		 * 
		 * @return a matrix corresponding with the meta file
		 * @throws IOException
		 *             If an I/O error occurs
		 * @throws MetaDataMatchException
		 *             if the meta data and dat data doesn't match
		 */
		protected static double[][] readMatrices(final LoadedGeneExpressionDatabase data)
				throws IOException, MetaDataMatchException {
			BufferedReader reader = findFile(data.path, DATA_SUFFIX);

			double[][] matrix = new double[data.examples][];

			for (int i = 0; i < data.examples; i++) {
				matrix[i] = (Arrays.stream(reader.readLine().split(SEPERATOR))
						.mapToDouble(val -> Math.min(1, Math.max(0, (Double.parseDouble(val) + 1) / 2))).toArray());
				if (matrix[i].length != data.width * data.height) {
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
		protected static int[] readLabels(final LoadedGeneExpressionDatabase data) throws IOException {
			BufferedReader reader = findFile(data.path, LABEL_SUFFIX);
			int[] label = new int[data.examples];
			for (int i = 0; i < data.examples; i++) {
				label[i] = (int) Double.parseDouble(reader.readLine());
			}
			reader.close();
			return label;
		}
	}
}
