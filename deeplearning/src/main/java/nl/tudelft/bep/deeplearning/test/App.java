package nl.tudelft.bep.deeplearning.test;

import java.io.IOException;

public class App {
	public static void main(String[] args) throws IOException, ClassNotFoundException {
		ResultUtil.generateCSV(10, new ComputeAverageAccuracyFiller(10), "results");
	}
}
