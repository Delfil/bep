package nl.tudelft.bep.deeplearning.network.result;

import java.util.stream.Collectors;

/**
 * A lister which lists all accuracies of the given network.
 */
public class ListAccuracy implements Lister {

	@Override
	public String list(final int epoch, final String dataSet, final String network) {
		return EvaluationFileUtil.load(epoch, dataSet, network).stream().map(e -> Double.toString(e.accuracy()))
				.collect(Collectors.joining("\n"));
	}

}
