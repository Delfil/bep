package nl.tudelft.bep.deeplearning.network.result;

import java.util.stream.Collectors;

public class ListAccurracy implements Lister {

	@Override
	public String list(int epoch, String dataSet, String network) {
		return EvaluationFileUtil.load(epoch, dataSet, network).stream().map(e -> Double.toString(e.accuracy()))
				.collect(Collectors.joining("\n"));
	}

}
