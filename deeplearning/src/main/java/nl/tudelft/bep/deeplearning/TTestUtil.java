package nl.tudelft.bep.deeplearning;

import java.util.Arrays;
import java.util.List;

import org.apache.commons.math3.stat.inference.TTest;
import org.deeplearning4j.eval.Evaluation;

import nl.tudelft.bep.deeplearning.datafetcher.DataPath;

public class TTestUtil {
	public static double get(String builder1, String data1, int epoch1, String builder2, String data2, int epoch2) {
		return get(FinishedNNCBuilder.load(builder1), DataPath.readDataSet(data1), epoch1,
				FinishedNNCBuilder.load(builder2), DataPath.readDataSet(data2), epoch2);
	}

	public static double get(FinishedNNCBuilder builder1, DataPath data1, int epoch1, FinishedNNCBuilder builder2,
			DataPath data2, int epoch2) {
		return get(FileUtil.load(epoch1, data1, builder1), FileUtil.load(epoch2, data2, builder2));
	}

	public static double get(List<Evaluation<Double>> sample1, List<Evaluation<Double>> sample2) {
		return get(getAccuracyArray(sample1), getAccuracyArray(sample2));
	}

	public static double get(double[] sample1, double[] sample2) {
		System.out.println(Arrays.toString(sample1));
		System.out.println();
		System.out.println(Arrays.toString(sample2));
		System.out.println();
		int min = Math.min(sample1.length, sample2.length);
		return new TTest().pairedTTest(Arrays.copyOf(sample1, min), Arrays.copyOf(sample2, min));
	}

	public static double[] getAccuracyArray(List<Evaluation<Double>> eval) {
		return eval.stream().mapToDouble(Evaluation::accuracy).toArray();
	}
}
