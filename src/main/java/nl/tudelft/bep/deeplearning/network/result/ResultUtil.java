package nl.tudelft.bep.deeplearning.network.result;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.StringJoiner;
import java.util.stream.Collectors;

import org.deeplearning4j.eval.Evaluation;

import nl.tudelft.bep.deeplearning.network.builder.FNNCBuilder;
import nl.tudelft.bep.deeplearning.network.data.Data;
import nl.tudelft.bep.deeplearning.network.result.csv.CSVFiller;
import nl.tudelft.bep.deeplearning.network.result.lister.Lister;

public final class ResultUtil {
	private static final int EXPECTED_DATA_FILE_COUNT = 3;

	/**
	 * Utility-classes should not be initialized.
	 */
	private ResultUtil() {
	}

	public static double[] getAccuracyArray(final List<Evaluation<Double>> sample) {
		return sample.stream().mapToDouble(Evaluation::accuracy).toArray();
	}

	public static double getAverageAccuracy(final String builder1, final String data, final int epoch) {
		return getAverageAccuracy(FNNCBuilder.load(builder1), Data.readDataSet(data), epoch);
	}

	public static double getAverageAccuracy(final FNNCBuilder builder, final Data data, final int epoch) {
		List<Evaluation<Double>> a = EvaluationFileUtil.load(epoch, data, builder);
		if (a == null || a.isEmpty()) {
			return Double.NaN;
		}
		return getAverageAccuracy(getAccuracyArray(a));
	}

	public static double getAverageAccuracy(final List<Evaluation<Double>> sample) {
		return getAverageAccuracy(getAccuracyArray(sample));
	}

	public static double getAverageAccuracy(final double[] sample) {
		return Arrays.stream(sample).reduce((a, b) -> a + b).getAsDouble() / sample.length;
	}

	/**
	 * Generate a csv file with data sets as columns and networks as rows,
	 * filled by the specified filler, and saves it on the disk.
	 * 
	 * @param epochs
	 *            the number of epochs on which the data should be filtered
	 * @param filler
	 *            the filler which should fill the table its cells.
	 */
	public static void generateCSV(final int epochs, final CSVFiller filler, final String fileName) {
		List<String> networkList = getNetworkList();
		List<String> dataList = getDataList();
		try {
			PrintWriter writer = new PrintWriter(new File(fileName + ".csv"));
			writer.write(new Date().toString());
			writer.write(",,");
			writer.write(dataList.stream().collect(Collectors.joining(",")));
			writer.write("\n");
			for (int y = 0; y < networkList.size(); y++) {
				StringJoiner sj = new StringJoiner(",");
				sj.add(networkList.get(y));
				sj.add(FNNCBuilder.getDescription(networkList.get(y)));
				for (int x = 0; x < dataList.size(); x++) {
					sj.add(filler.fill(networkList.get(y), dataList.get(x), epochs));
				}
				writer.write(sj.toString());
				writer.write("\n");
				writer.flush();
			}
			writer.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Get all network file names from the network folder.
	 * 
	 * @return a list of file names
	 */
	public static List<String> getNetworkList() {
		List<String> networkList = new ArrayList<>();
		for (File file : new File(FNNCBuilder.NETWORK_FOLDER).listFiles()) {
			if (file.isDirectory()) {
				for (File file2 : file.listFiles()) {
					if (file2.isFile() && file2.getName().startsWith(FNNCBuilder.MULTI_LAYER_NETWORK)
							&& file2.getName().endsWith(FNNCBuilder.NETWORK_SUFFIX)) {
						networkList.add(file2.getPath());
					}
				}
			}
		}
		return networkList;
	}

	/**
	 * Get all data set file names from the network folder.
	 * 
	 * @return a list of file names
	 */
	public static List<String> getDataList() {
		List<String> dataList = new ArrayList<>();
		for (File file : new File(Data.DATA_FOLDER).listFiles()) {
			if (file.isDirectory()) {
				System.out.println(file.getName());
				List<String> list = Arrays.stream(file.listFiles()).filter(File::isFile).map(File::getName)
						.map(s -> s.split("\\.")).filter(a -> a.length > 0).map(a -> a[a.length - 1])
						.filter(a -> a.equals("meta") || a.equals("dat") || a.equals("lab"))
						.collect(Collectors.toList());
				if (list.contains("meta") && list.contains("dat") && list.contains("lab")
						&& list.size() == EXPECTED_DATA_FILE_COUNT) {
					String[] path = file.getName().split("/");
					dataList.add(path[path.length - 1]);
				}
			}
		}
		return dataList;
	}

	/**
	 * Generate lists with the given lister, for each given data set network
	 * combination.
	 * 
	 * @param epoch
	 *            the number of epochs used
	 * @param lister
	 *            the lister to use
	 * @param pathName
	 *            the path name to save the lists in
	 * @param networkList
	 *            the list of networks to use
	 * @param dataList
	 *            the list of data sets to use
	 */
	public static void generateLists(final int epoch, final Lister lister, final String pathName,
			final List<String> networkList, final List<String> dataList) {
		try {
			for (int y = 0; y < networkList.size(); y++) {
				for (int x = 0; x < dataList.size(); x++) {
					String path = pathName + "/" + dataList.get(x) + "/" + networkList.get(y);
					
					new File(new File(path).getParent()).mkdirs();

					PrintWriter writer = new PrintWriter(path + ".csv", "UTF-8");
					writer.write(FNNCBuilder.getDescription(networkList.get(y)));
					writer.write("\n");
					writer.write(networkList.get(y));
					writer.write("\n");
					writer.write(dataList.get(x));
					writer.write("\n");
					writer.write(lister.list(epoch, dataList.get(x), networkList.get(y)));
					writer.close();
				}
			}
		} catch (FileNotFoundException | UnsupportedEncodingException e) {
			e.printStackTrace();
		}
	}

	/**
	 * Generate lists with the given lister, for each existing data set network
	 * combination.
	 * 
	 * @param epoch
	 *            the number of epochs used
	 * @param lister
	 *            the lister to use
	 * @param pathName
	 *            the path name to save the lists in
	 */
	public static void generateLists(final int epoch, final Lister lister, final String pathName) {
		generateLists(epoch, lister, pathName, getNetworkList(), getDataList());
	}
}
