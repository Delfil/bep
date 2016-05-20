package nl.tudelft.bep.deeplearning.network.result;

import java.io.IOException;

import nl.tudelft.bep.deeplearning.network.result.csv.ComputeAverageAccuracyFiller;

public class App {
	public static void main(String[] args) throws IOException, ClassNotFoundException {
		ResultUtil.generateCSV(10, new ComputeAverageAccuracyFiller(10), "results");
	}
}
