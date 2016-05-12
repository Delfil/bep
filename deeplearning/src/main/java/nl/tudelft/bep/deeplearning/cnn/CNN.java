package nl.tudelft.bep.deeplearning.cnn;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.weights.WeightInit;

import nl.tudelft.bep.deeplearning.NNCBuilder;

public class CNN {
	public static NNCBuilder BuildExampleCNN(Layer... layers){
		NNCBuilder builder = new NNCBuilder();
				builder
				// Initialization
				.weightInit(WeightInit.XAVIER)
				.seed(0)
				// Training
				.iterations(1)
				.learningRate(0.01)
				.momentum(0.9)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(Updater.NESTEROVS)
				
				// Over fitting
				.l2(0.0005)
				.regularization(true);
		
		for(int i = 0 ; i < layers.length; i++) {
			builder.add(layers[i]);
		}
		builder.backprop(true).pretrain(false);
		

		return builder;
	}
}
