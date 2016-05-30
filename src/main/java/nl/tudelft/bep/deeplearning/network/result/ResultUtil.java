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

import org.apache.commons.math3.stat.inference.TTest;
import org.deeplearning4j.eval.Evaluation;

import nl.tudelft.bep.deeplearning.network.builder.FNNCBuilder;
import nl.tudelft.bep.deeplearning.network.data.Data;
import nl.tudelft.bep.deeplearning.network.result.csv.CSVFiller;

public final class ResultUtil {
	private static final int EXPECTED_DATA_FILE_COUNT = 3;

	/**
	 * Utility-classes should not be initialized.
	 */
	private ResultUtil() {
	}

	public static double getTTest(final String builder1, final String data1, final int epoch1, final String builder2,
			final String data2, final int epoch2) {
		return getTTest(FNNCBuilder.load(builder1), Data.readDataSet(data1), epoch1, FNNCBuilder.load(builder2),
				Data.readDataSet(data2), epoch2);
	}

	public static double getTTest(final FNNCBuilder builder1, final Data data1, final int epoch1,
			final FNNCBuilder builder2, final Data data2, final int epoch2) {
		return getTTest(EvaluationFileUtil.load(epoch1, data1, builder1),
				EvaluationFileUtil.load(epoch2, data2, builder2));
	}

	public static double getTTest(final List<Evaluation<Double>> sample1, final List<Evaluation<Double>> sample2) {
		return getTTest(getAccuracyArray(sample1), getAccuracyArray(sample2));
	}

	public static double getTTest(final double[] sample1, final double[] sample2) {
		int min = Math.min(sample1.length, sample2.length);
		return new TTest().pairedTTest(Arrays.copyOf(sample1, min), Arrays.copyOf(sample2, min));
	}

	public static double getTTest(final String builder1, final String data1, final int epoch1) {
		return getTTest(FNNCBuilder.load(builder1), Data.readDataSet(data1), epoch1);
	}

	public static double getTTest(final FNNCBuilder builder1, final Data data1, final int epoch1) {
		return getTTest(EvaluationFileUtil.load(epoch1, data1, builder1));
	}

	public static double getTTest(final List<Evaluation<Double>> sample1) {
		return getTTest(getAccuracyArray(sample1));
	}

	public static double getTTest(final double[] sample1) {
		int min = sample1.length / 2;
		return new TTest().tTest(Arrays.copyOf(sample1, min), Arrays.copyOfRange(sample1, min, min * 2));
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

	public static void generateLists(int epoch, Lister lister, String pathName) {
		List<String> networkList = getNetworkList();
		List<String> dataList = getDataList();
		try {
			for (int y = 0; y < networkList.size(); y++) {
				for (int x = 0; x < dataList.size(); x++) {
					String path = pathName + "/" + dataList.get(x) + "/" + networkList.get(y);
					String[] split = path.split("/");
					new File(path.substring(0, path.length() - split[split.length - 1].length())).mkdirs();
					
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
}
