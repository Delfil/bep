package nl.tudelft.bep.deeplearning;

import java.io.IOException;

import nl.tudelft.bep.deeplearning.network.result.ResultUtil;
import nl.tudelft.bep.deeplearning.network.result.csv.ComputeAverageAccuracyFiller;

public final class App {
	private static final int DEFAULT_EPOCHS = 10;
	private static final int DEFAULT_ITERATIONS = 20;
	private static final String DEFAULT_FILE_NAME = "results";

	private App() {
	}

	public static void main(final String[] args) throws IOException, ClassNotFoundException {
		run((args.length >= 1 && isInteger(args[0])) ? Integer.parseInt(args[0]) : DEFAULT_EPOCHS,
				(args.length >= 2 && isInteger(args[1])) ? Integer.parseInt(args[1]) : DEFAULT_ITERATIONS,
				(args.length >= 3 ? args[2] : DEFAULT_FILE_NAME));
	}

	private static boolean isInteger(final String string) {
		return string.matches("-?\\d+?");
	}

	private static void run(final int epochs, final int iterations, final String fileName) {
		ResultUtil.generateCSV(epochs, new ComputeAverageAccuracyFiller(iterations), fileName);
	}
}
