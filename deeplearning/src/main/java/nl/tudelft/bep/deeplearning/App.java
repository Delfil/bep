package nl.tudelft.bep.deeplearning;

import java.io.IOException;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class App {

	public static void main(String[] args) throws IOException {
		int batchSize = 64;
		int epoch = 2;
		int iter = 100;

		DataSetIterator trainData = new MnistDataSetIterator(batchSize, true, 1);
		DataSetIterator testData = new MnistDataSetIterator(batchSize, false, 34);

		MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder().iterations(iter).seed(100)
				.learningRate(0.1).weightInit(WeightInit.NORMALIZED)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).list(6)
				.layer(0,
						new ConvolutionLayer.Builder(5, 5).nIn(1).stride(1, 1).nOut(20).activation("identity").build())
				.layer(1,
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build())
				.layer(2,
						new ConvolutionLayer.Builder(5, 5).nIn(1).stride(1, 1).nOut(50).activation("identity").build())
				.layer(3,
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build())
				.layer(4, new DenseLayer.Builder().activation("relu").nOut(500).build())
				.layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(10)
						.activation("softmax").build())
				.backpropType(BackpropType.Standard).backprop(true);

		new ConvolutionLayerSetup(builder, 28, 28, 1);

		MultiLayerConfiguration conf = builder.build();
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();

		model.setListeners(new ScoreIterationListener(10));
		for (int i = 0; i < epoch; i++) {
			model.fit(trainData);
			Evaluation eval = new Evaluation(10);
			while (testData.hasNext()) {
				DataSet temp = testData.next();
				INDArray output = model.output(temp.getFeatureMatrix());
				eval.eval(temp.getLabels(), output);
			}
			trainData.reset();
		}
	}
}
