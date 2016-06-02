package nl.tudelft.bep.deeplearning.network.builder;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.weights.WeightInit;

public abstract class CNN {
	private static final double MOMENTUM = 0.9;
	private static final double LEARNING_RATE = 0.01;
	private static final double L2 = 0.0005;

	public static NNCBuilder buildExampleCNN(final Layer... layers) {
		NNCBuilder builder = new NNCBuilder();
		builder
				// Initialization
				.weightInit(WeightInit.XAVIER).seed(0)
				// Training
				.iterations(1).learningRate(LEARNING_RATE).momentum(MOMENTUM)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.NESTEROVS)

				// Over fitting
				.l2(L2).regularization(true).iterations(20);

		for (Layer layer : layers) {
			builder.add(layer);
		}
		builder.backprop(true).pretrain(false);

		return builder;
	}
}
