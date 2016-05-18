package nl.tudelft.bep.deeplearning.test;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.StringJoiner;
import java.util.stream.Collectors;

import org.apache.commons.math3.stat.inference.TTest;
import org.deeplearning4j.eval.Evaluation;

import nl.tudelft.bep.deeplearning.data.Data;
import nl.tudelft.bep.deeplearning.network.FinishedNNCBuilder;

public class ResultUtil {
	public static double getTTest(String builder1, String data1, int epoch1, String builder2, String data2,
			int epoch2) {
		return getTTest(FinishedNNCBuilder.load(builder1), Data.readDataSet(data1), epoch1,
				FinishedNNCBuilder.load(builder2), Data.readDataSet(data2), epoch2);
	}

	public static double getTTest(FinishedNNCBuilder builder1, Data data1, int epoch1, FinishedNNCBuilder builder2,
			Data data2, int epoch2) {
		return get(EvaluationFileUtil.load(epoch1, data1, builder1), EvaluationFileUtil.load(epoch2, data2, builder2));
	}

	public static double get(List<Evaluation<Double>> sample1, List<Evaluation<Double>> sample2) {
		return get(getAccuracyArray(sample1), getAccuracyArray(sample2));
	}

	public static double get(double[] sample1, double[] sample2) {
		int min = Math.min(sample1.length, sample2.length);
		return new TTest().pairedTTest(Arrays.copyOf(sample1, min), Arrays.copyOf(sample2, min));
	}

	public static double[] getAccuracyArray(List<Evaluation<Double>> sample) {
		return sample.stream().mapToDouble(Evaluation::accuracy).toArray();
	}

	public static double getAverageAccuracy(String builder1, String data, int epoch) {
		return getAverageAccuracy(FinishedNNCBuilder.load(builder1), Data.readDataSet(data), epoch);
	}

	public static double getAverageAccuracy(FinishedNNCBuilder builder, Data data, int epoch) {
		List<Evaluation<Double>> a = EvaluationFileUtil.load(epoch, data, builder);
		if (a == null || a.isEmpty()) {
			return Double.NaN;
		}
		return getAverageAccuracy(getAccuracyArray(a));
	}

	public static double getAverageAccuracy(List<Evaluation<Double>> sample) {
		return getAverageAccuracy(getAccuracyArray(sample));
	}

	public static double getAverageAccuracy(double[] sample) {
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
	public static void generateCSV(int epochs, CSVFiller filler) {
		List<String> networkList = getNetworkList();
		List<String> dataList = getDataList();
		try {
			PrintWriter writer = new PrintWriter(new File("results.csv"));
			writer.write(new Date().toString());
			writer.write(",");
			writer.write(dataList.stream().collect(Collectors.joining(",")));
			writer.write("\n");
			for (int y = 0; y < networkList.size(); y++) {
				StringJoiner sj = new StringJoiner(",");
				sj.add(networkList.get(y));
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
	private static List<String> getNetworkList() {
		List<String> networkList = new ArrayList<>();
		for (File file : new File(FinishedNNCBuilder.NETWORK_FOLDER).listFiles()) {
			if (file.isDirectory()) {
				for (File file2 : file.listFiles()) {
					if (file2.isFile()) {
						if (file2.getName().endsWith(FinishedNNCBuilder.NETWORK_SUFFIX)) {
							networkList.add(file2.getPath());
						}
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
	protected static List<String> getDataList() {
		List<String> dataList = new ArrayList<>();
		for (File file : new File(Data.DATA_FOLDER).listFiles()) {
			if (file.isDirectory()) {
				String[] path = file.getName().split("/");
				dataList.add(path[path.length - 1]);
			}
		}
		return dataList;
	}
}
