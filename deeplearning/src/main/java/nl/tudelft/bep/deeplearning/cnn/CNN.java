package nl.tudelft.bep.deeplearning.cnn;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.weights.WeightInit;

import nl.tudelft.bep.deeplearning.NNConfigurationBuilder;

public class CNN {
	public static NNConfigurationBuilder BuildExampleCNN(Layer... layers){
		NNConfigurationBuilder builder = new NNConfigurationBuilder();
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
//				
//				// layers
//				.list(layers.length);
		
		for(int i = 0 ; i < layers.length; i++) {
			builder.add(layers[i]);
		}
		builder.backprop(true).pretrain(false);
		

		return builder;
		
//		new ConvolutionLayerSetup(builder, width, height, 1);		
//		MultiLayerConfiguration conf = builder.build();
//		System.out.println(conf.toJson());
//		MultiLayerNetwork model = new MultiLayerNetwork(conf);
//		model.init();
//		return model;
	}
}
